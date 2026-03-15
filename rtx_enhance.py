#!/usr/bin/env python3
"""
rtx_enhance — Video enhancement pipeline combining:
  1. DeH264-RTMoSR (H.264 artifact removal / decompression)
  2. Nvidia RTX Video Super Resolution (nvvfx upscaling)

Usage:
  python rtx_enhance.py input.mp4 output.mp4 [options]

Requires: nvidia-vfx, torch, ffmpeg in PATH
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

# ─── DeH264 model loading ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# Preset name -> (filename, description, loader)
DEH264_PRESETS = {
    "fast": (
        "1xDeH264_RTMoSR.pth",
        "RTMoSR (Fast) — good balance of speed and quality for high/medium sources",
        "rtmosr",
    ),
    "ultrafast": (
        "deH264_SuperUltraCompact.safetensors",
        "SuperUltraCompact (UltraFast) — fastest, best for high/medium quality sources",
        "spandrel",
    ),
    "best": (
        "1xDeH264_realplksr.pth",
        "RealPLKSR (Very Slow) — strongest artifact removal for low quality sources",
        "spandrel",
    ),
}

DEFAULT_PRESET = "fast"


def load_deh264_model(
    model_path: str, loader: str = "auto", device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a DeH264 artifact removal model.
    loader: 'rtmosr' for RTMoSR arch, 'spandrel' for auto-detect, 'auto' to guess
    """
    model_path = str(model_path)

    # Auto-detect loader based on filename
    if loader == "auto":
        if "RTMoSR" in model_path and "Unshuffle" not in model_path:
            loader = "rtmosr"
        else:
            loader = "spandrel"

    if loader == "rtmosr":
        from rtmosr import RTMoSR

        model = RTMoSR(scale=1, dim=32, ffn_expansion=2, n_blocks=2, se=True)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        model.load_state_dict(state_dict)
        model.eval().to(device)
        return model
    else:
        import spandrel

        model_desc = spandrel.ModelLoader(device=device).load_from_file(model_path)
        model_desc.model.eval()
        return model_desc.model


@torch.inference_mode()
def deh264_process_frame(
    model: torch.nn.Module, frame: np.ndarray, tile_size: int = 0, device: str = "cuda"
) -> np.ndarray:
    """
    Run DeH264-RTMoSR on a single frame.
    frame: HxWx3 uint8 RGB numpy array
    Returns: HxWx3 uint8 RGB numpy array
    """
    # Convert to tensor [1, 3, H, W] float32 [0, 1]
    tensor = torch.from_numpy(frame).to(device, non_blocking=True).float().div_(255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    if tile_size > 0:
        output = _tiled_inference(model, tensor, tile_size)
    else:
        output = model(tensor)

    output = output.squeeze(0).permute(1, 2, 0).clamp_(0, 1).mul_(255.0)
    return output.byte().cpu().numpy()


def _tiled_inference(
    model: torch.nn.Module, tensor: torch.Tensor, tile_size: int, overlap: int = 16
) -> torch.Tensor:
    """Process a frame in tiles to save VRAM."""
    _, _, h, w = tensor.shape
    out = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)

    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = tensor[:, :, y_start:y_end, x_start:x_end]
            tile_out = model(tile)

            out[:, :, y_start:y_end, x_start:x_end] += tile_out
            weight[:, :, y_start:y_end, x_start:x_end] += 1.0

    return out / weight


# ─── RTX Video Super Resolution (nvvfx) ────────────────────────────────────


def create_rtx_upscaler(output_width: int, output_height: int, quality: str = "ULTRA"):
    """Create and configure the nvvfx VideoSuperRes context."""
    import nvvfx

    quality_map = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    q = quality_map.get(quality.upper(), nvvfx.effects.QualityLevel.ULTRA)
    sr = nvvfx.VideoSuperRes(q)
    sr.output_width = output_width
    sr.output_height = output_height
    sr.load()
    return sr


@torch.inference_mode()
def rtx_upscale_frame(sr, frame: np.ndarray) -> np.ndarray:
    """
    Upscale a single frame using RTX Video Super Resolution.
    frame: HxWx3 uint8 RGB numpy array
    Returns: upscaled HxWx3 uint8 RGB numpy array
    """
    # nvvfx expects [3, H, W] float32 [0, 1] on CUDA
    tensor = torch.from_numpy(frame).cuda(non_blocking=True).float().div_(255.0)
    tensor = tensor.permute(2, 0, 1).contiguous()

    dlpack_out = sr.run(tensor).image
    output = torch.from_dlpack(dlpack_out)

    # output is [3, H_out, W_out] — skip .clone(), go straight to bytes
    output = output.permute(1, 2, 0).clamp_(0, 1).mul_(255.0)
    return output.byte().cpu().numpy()


# ─── ffmpeg video I/O ──────────────────────────────────────────────────────


def probe_video(input_path: str) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = None
    audio_streams = []
    for s in info.get("streams", []):
        if s["codec_type"] == "video" and video_stream is None:
            video_stream = s
        elif s["codec_type"] == "audio":
            audio_streams.append(s)

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {input_path}")

    # Parse framerate
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den if den else 30.0

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "fps_str": r_frame_rate,
        "total_frames": int(video_stream.get("nb_frames", 0)) or None,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "has_audio": len(audio_streams) > 0,
        "pix_fmt": video_stream.get("pix_fmt", "yuv420p"),
        "codec": video_stream.get("codec_name", "unknown"),
    }


def open_video_reader(input_path: str, width: int, height: int) -> subprocess.Popen:
    """Open ffmpeg process to decode video frames as raw RGB, with hwaccel if available."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "auto",
        "-i",
        input_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "quiet",
        "-",
    ]
    # 8 frames of read-ahead buffer
    buf = width * height * 3 * 8
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=buf)


def open_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps_str: str,
    input_path: str,
    codec: str,
    crf: int,
    preset: str,
    extra_args: list[str] | None = None,
    copy_audio: bool = True,
) -> subprocess.Popen:
    """Open ffmpeg process to encode video from raw RGB input."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # Raw RGB input from pipe
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_str,
        "-i",
        "-",
    ]

    # Audio passthrough from original file
    if copy_audio:
        cmd += ["-i", input_path, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]

    # Video codec settings
    codec_args = _get_codec_args(codec, crf, preset)
    cmd += codec_args

    # Extra user-provided ffmpeg args
    if extra_args:
        cmd += extra_args

    cmd += [output_path]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=width * height * 3 * 4)


def _get_codec_args(codec: str, crf: int, preset: str) -> list[str]:
    """Return ffmpeg codec arguments for the chosen output codec."""
    codec = codec.lower()

    if codec in ("h264", "libx264", "x264"):
        return [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]
    elif codec in ("h265", "hevc", "libx265", "x265"):
        return [
            "-c:v",
            "libx265",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
        ]
    elif codec in ("av1", "svtav1", "libsvtav1"):
        import os

        return [
            "-c:v",
            "libsvtav1",
            "-preset",
            str(min(8, max(0, int(preset) if preset.isdigit() else 4))),
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p10le",
            "-svtav1-params",
            "tune=0:lp=6",
        ]
    elif codec in ("prores", "prores_ks"):
        # ProRes doesn't use CRF — profile 3 = ProRes HQ
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    elif codec in ("nvenc", "h264_nvenc"):
        return [
            "-c:v",
            "h264_nvenc",
            "-cq:v",
            str(crf),
            "-preset",
            "p6",
            "-pix_fmt",
            "yuv420p10le",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-aq-strength",
            "8",
        ]
    elif codec in ("hevc_nvenc", "h265_nvenc"):
        return [
            "-c:v",
            "hevc_nvenc",
            "-cq:v",
            str(crf),
            "-preset",
            "p6",
            "-pix_fmt",
            "yuv420p10le",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-aq-strength",
            "8",
            "-tag:v",
            "hvc1",
        ]
    elif codec in ("av1_nvenc",):
        return [
            "-c:v",
            "av1_nvenc",
            "-cq:v",
            str(crf),
            "-preset",
            "p6",
            "-pix_fmt",
            "yuv420p10le",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-aq-strength",
            "8",
        ]
    elif codec == "ffv1":
        return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p"]
    elif codec in ("utvideo",):
        return ["-c:v", "utvideo", "-pix_fmt", "rgb24"]
    else:
        # Pass through as-is (user knows what they're doing)
        return ["-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]


# ─── Helpers ───────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    else:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h}h {m:02d}m"


# ─── Main pipeline ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RTX Video Enhancement: DeH264 artifact removal + RTX Super Resolution upscaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2x upscale (default — just upscaling, no decompression)
  python rtx_enhance.py input.mp4 output.mkv

  # Decompress + 2x upscale
  python rtx_enhance.py input.mp4 output.mkv --decompress

  # Maximum quality: best decompression + 4x upscale, HEVC output
  python rtx_enhance.py input.mp4 output.mkv --decompress --deh264-preset best --scale 4 --codec hevc

  # Fastest decompression + upscale (good for batch processing)
  python rtx_enhance.py input.mp4 output.mkv --decompress --deh264-preset ultrafast --rtx-quality LOW

  # Decompress only (no upscaling) — clean up a crusty source
  python rtx_enhance.py input.mp4 output.mkv --decompress --no-upscale --deh264-preset best

  # Target specific output resolution
  python rtx_enhance.py input.mp4 output_4k.mkv --target-width 3840 --target-height 2160

  # ProRes output for editing
  python rtx_enhance.py input.mp4 output.mov --codec prores

  # Use NVENC hardware encoding
  python rtx_enhance.py input.mp4 output.mkv --codec nvenc --crf 20
        """,
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")

    # Pipeline stages
    stage = parser.add_argument_group("Pipeline stages")
    stage.add_argument(
        "--decompress",
        action="store_true",
        help="Enable DeH264 artifact removal before upscaling",
    )
    stage.add_argument(
        "--no-upscale",
        action="store_true",
        help="Skip RTX upscaling (decompress only, requires --decompress)",
    )

    # DeH264 options
    deh = parser.add_argument_group("DeH264 artifact removal")
    preset_help = "DeH264 quality preset:\n"
    for name, (fname, desc, _) in DEH264_PRESETS.items():
        preset_help += f"  {name:10s} — {desc}\n"
    deh.add_argument(
        "--deh264-preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list(DEH264_PRESETS.keys()),
        help=f"DeH264 quality/speed preset (default: {DEFAULT_PRESET})",
    )
    deh.add_argument(
        "--deh264-model",
        type=str,
        default=None,
        help="Path to a custom DeH264 model (overrides --deh264-preset)",
    )
    deh.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Process DeH264 in tiles of this size (0 = full frame, use if OOM)",
    )

    # RTX upscale options
    rtx = parser.add_argument_group("RTX upscaling")
    rtx.add_argument(
        "--scale", type=float, default=2.0, help="Upscale factor (default: 2.0)"
    )
    rtx.add_argument(
        "--target-width",
        type=int,
        default=0,
        help="Target output width (overrides --scale)",
    )
    rtx.add_argument(
        "--target-height",
        type=int,
        default=0,
        help="Target output height (overrides --scale)",
    )
    rtx.add_argument(
        "--rtx-quality",
        type=str,
        default="ULTRA",
        choices=["LOW", "MEDIUM", "HIGH", "ULTRA"],
        help="RTX VSR quality level (default: ULTRA)",
    )

    # Output encoding
    enc = parser.add_argument_group("Output encoding")
    enc.add_argument(
        "--codec",
        type=str,
        default="av1",
        help="Output codec: av1, av1_nvenc, hevc_nvenc, nvenc, h264, h265/hevc, prores, ffv1, utvideo (default: av1)",
    )
    enc.add_argument(
        "--crf", type=int, default=25, help="CRF/CQ quality (lower=better, default: 25)"
    )
    enc.add_argument(
        "--preset",
        type=str,
        default="4",
        help="Encoder preset (default: 4). For av1: 0-8, for h264/h265: ultrafast-veryslow",
    )
    enc.add_argument(
        "--no-audio", action="store_true", help="Don't copy audio from input"
    )
    enc.add_argument(
        "--ffmpeg-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra ffmpeg output arguments (passed directly)",
    )

    args = parser.parse_args()

    if not args.decompress and args.no_upscale:
        print(
            "Error: Nothing to do — --no-upscale requires --decompress",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── Probe input ─────────────────────────────────────────────────────
    print(f"Probing {args.input}...")
    info = probe_video(args.input)
    in_w, in_h = info["width"], info["height"]
    print(
        f"  Source: {in_w}x{in_h} @ {info['fps']:.3f}fps, codec={info['codec']}, "
        f"duration={info['duration']:.1f}s"
    )

    if info["total_frames"]:
        print(f"  Frames: {info['total_frames']}")

    # ── Compute output dimensions ───────────────────────────────────────
    if args.no_upscale:
        out_w, out_h = in_w, in_h
    elif args.target_width > 0 and args.target_height > 0:
        out_w = args.target_width
        out_h = args.target_height
    else:
        out_w = int(in_w * args.scale)
        out_h = int(in_h * args.scale)

    # RTX requires dimensions divisible by 8
    out_w = max(8, (out_w // 8) * 8)
    out_h = max(8, (out_h // 8) * 8)

    print(f"  Output: {out_w}x{out_h}")

    # ── Load models ─────────────────────────────────────────────────────
    deh264_model = None
    rtx_sr = None

    if args.decompress:
        # Resolve model path from preset or explicit path
        if args.deh264_model:
            model_path = args.deh264_model
            loader = "auto"
            preset_name = "custom"
        else:
            fname, desc, loader = DEH264_PRESETS[args.deh264_preset]
            model_path = str(MODELS_DIR / fname)
            preset_name = args.deh264_preset

        if not os.path.isfile(model_path):
            print(f"Error: DeH264 model not found: {model_path}", file=sys.stderr)
            print(f"  Download it to {MODELS_DIR}/", file=sys.stderr)
            sys.exit(1)

        print(f"Loading DeH264 model [{preset_name}] from {model_path}...")
        deh264_model = load_deh264_model(model_path, loader=loader)
        print("  DeH264 model loaded.")

    if not args.no_upscale:
        print(f"Initializing RTX Video Super Resolution ({args.rtx_quality})...")
        import nvvfx

        rtx_sr = create_rtx_upscaler(out_w, out_h, args.rtx_quality)
        print(f"  RTX VSR loaded: {in_w}x{in_h} -> {out_w}x{out_h}")

    # ── Open ffmpeg reader/writer ───────────────────────────────────────
    print("Starting video processing...")
    reader = open_video_reader(args.input, in_w, in_h)
    writer = open_video_writer(
        args.output,
        out_w,
        out_h,
        info["fps_str"],
        args.input,
        args.codec,
        args.crf,
        args.preset,
        extra_args=args.ffmpeg_args,
        copy_audio=not args.no_audio and info["has_audio"],
    )

    frame_bytes = in_w * in_h * 3
    SENTINEL = None  # signals end-of-stream
    QUEUE_DEPTH = 32  # frames buffered between stages
    error_event = threading.Event()  # signals any thread hit an error

    # Shared counters (updated by writer thread, read by progress)
    frame_count = 0
    t_start = time.time()

    # ── Pre-allocate pinned memory for fast CPU→GPU transfers ─────
    pinned_buffer = torch.empty(in_h, in_w, 3, dtype=torch.uint8, pin_memory=True)

    # ── Decoder thread ──────────────────────────────────────────────
    decode_q = queue.Queue(maxsize=QUEUE_DEPTH)

    def decoder_thread():
        """Read raw frames from ffmpeg, upload to GPU, push CUDA tensor."""
        try:
            stream = torch.cuda.Stream()
            while not error_event.is_set():
                raw = reader.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                # Copy into pinned memory (fast) then async upload to GPU
                np.copyto(
                    pinned_buffer.numpy(),
                    np.frombuffer(raw, dtype=np.uint8).reshape(in_h, in_w, 3),
                )
                with torch.cuda.stream(stream):
                    gpu_tensor = (
                        pinned_buffer.to("cuda", non_blocking=True).float().div_(255.0)
                    )
                    gpu_tensor = gpu_tensor.permute(2, 0, 1).contiguous()
                stream.synchronize()
                decode_q.put(gpu_tensor)
        except Exception as e:
            error_event.set()
            print(f"\n  Decoder error: {e}", file=sys.stderr)
        finally:
            decode_q.put(SENTINEL)

    # ── GPU processing thread ───────────────────────────────────────
    gpu_q = queue.Queue(maxsize=QUEUE_DEPTH)

    def gpu_thread():
        """Process CUDA tensors through DeH264 and/or RTX VSR, output numpy."""
        try:
            while not error_event.is_set():
                tensor = decode_q.get()
                if tensor is SENTINEL:
                    break

                with torch.inference_mode():
                    # Stage 1: DeH264 artifact removal
                    if deh264_model is not None:
                        t = tensor.unsqueeze(0)  # [1, 3, H, W]
                        if args.tile_size > 0:
                            t = _tiled_inference(deh264_model, t, args.tile_size)
                        else:
                            t = deh264_model(t)
                        tensor = t.squeeze(0).clamp_(0, 1)  # [3, H, W]

                    # Stage 2: RTX upscaling
                    if rtx_sr is not None:
                        dlpack_out = rtx_sr.run(tensor).image
                        tensor = torch.from_dlpack(dlpack_out)

                    # Convert to output bytes on GPU then transfer once
                    frame = (
                        tensor.permute(1, 2, 0)
                        .contiguous()
                        .clamp_(0, 1)
                        .mul_(255.0)
                        .byte()
                        .cpu()
                        .numpy()
                    )

                gpu_q.put(frame)
        except Exception as e:
            error_event.set()
            print(f"\n  GPU error: {e}", file=sys.stderr)
        finally:
            gpu_q.put(SENTINEL)

    # ── Encoder thread ──────────────────────────────────────────────
    def encoder_thread():
        """Write processed frames to ffmpeg encoder."""
        nonlocal frame_count
        try:
            while not error_event.is_set():
                frame = gpu_q.get()
                if frame is SENTINEL:
                    break
                # Use memoryview to avoid an extra copy in .tobytes()
                writer.stdin.write(memoryview(frame))
                frame_count += 1
        except Exception as e:
            error_event.set()
            print(f"\n  Encoder error: {e}", file=sys.stderr)

    # ── Start all threads ───────────────────────────────────────────
    threads = [
        threading.Thread(target=decoder_thread, name="decoder", daemon=True),
        threading.Thread(target=gpu_thread, name="gpu", daemon=True),
        threading.Thread(target=encoder_thread, name="encoder", daemon=True),
    ]
    for t in threads:
        t.start()

    # ── Progress display (main thread) ──────────────────────────────
    total = info["total_frames"]
    if not total and info["duration"] > 0:
        total = int(info["duration"] * info["fps"])
        total_is_estimate = True
    else:
        total_is_estimate = False

    try:
        while threads[2].is_alive():
            threads[2].join(timeout=0.5)
            if frame_count == 0:
                continue
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            if total:
                pct = frame_count / total * 100
                remaining = (total - frame_count) / fps if fps > 0 else 0
                eta_time = datetime.now() + timedelta(seconds=remaining)
                prefix = "~" if total_is_estimate else ""
                print(
                    f"\r  Frame {frame_count}/{prefix}{total} "
                    f"({pct:.1f}%) | {fps:.2f} fps | "
                    f"{_fmt_duration(remaining)} remaining | "
                    f"done ~{eta_time.strftime('%H:%M')}    ",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r  Frame {frame_count} | {fps:.2f} fps    ",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        error_event.set()

    # ── Cleanup ─────────────────────────────────────────────────────
    for t in threads:
        t.join(timeout=5)

    reader.stdout.close()
    reader.wait()
    writer.stdin.close()
    writer.wait()

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print(
        f"\n\nDone! Processed {frame_count} frames in {_fmt_duration(elapsed)} ({avg_fps:.2f} fps)"
    )
    print(f"  Output: {args.output}")

    if os.path.isfile(args.output):
        out_size = os.path.getsize(args.output)
        print(f"  File size: {out_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
