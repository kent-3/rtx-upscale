#!/usr/bin/env python3
"""
compare — Generate side-by-side comparison videos: Bicubic vs RTX Video SR.

Upscales the full source frame with both bicubic and RTX Video SR, then
crops the same region from both to produce a labeled side-by-side comparison.
Cropping after upscaling gives RTX SR full-frame context for best results.

The pipeline:
  1. Trim source to time range (optional)
  2. Bicubic upscale full frame
  3. RTX SR upscale full frame (via rtx_enhance.py or existing file)
  4. Crop the same region from both upscaled outputs
  5. hstack with labels -> comparison video

Usage:
  # 2x upscale, crop 960x1080 for a 1920x1080 comparison
  python compare.py source_540p.mp4 --crop 960x1080+480+0

  # With existing RTX SR file
  python compare.py source.mp4 --rtxsr upscaled.mp4 --crop 960x1080+480+0

  # Short clip, Twitter-ready
  python compare.py source.mp4 --crop 960x1080+200+0 --start 5 --duration 6 --twitter

Requires: ffmpeg in PATH
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# ─── Default fonts ─────────────────────────────────────────────────────────

FONT_TITLE = os.path.expanduser("~/.local/share/fonts/Montserrat-BlackItalic.ttf")
FONT_SUB = os.path.expanduser("~/.local/share/fonts/Montserrat-Medium.ttf")


# ─── Video probing ─────────────────────────────────────────────────────────


def probe_video(path: str) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = None
    for s in info.get("streams", []):
        if s["codec_type"] == "video" and video_stream is None:
            video_stream = s

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {path}")

    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den if den else 30.0

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "fps_str": r_frame_rate,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
    }


# ─── Crop geometry parsing ─────────────────────────────────────────────────


def parse_crop(crop: str) -> tuple[int, int, int, int]:
    """
    Parse crop string 'WxH+X+Y' into (w, h, x, y).
    Also accepts 'W:H:X:Y' ffmpeg-style.
    """
    normalized = crop.replace("x", ":").replace("+", ":")
    parts = normalized.split(":")
    if len(parts) != 4:
        print(
            f"Error: invalid crop '{crop}' — expected WxH+X+Y or W:H:X:Y",
            file=sys.stderr,
        )
        sys.exit(1)
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


# ─── ffmpeg helpers ────────────────────────────────────────────────────────


def run_ffmpeg(args: list[str], desc: str = "") -> None:
    """Run an ffmpeg command, printing what we're doing."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args
    if desc:
        print(f"  {desc}")
        print(f"    $ ffmpeg {' '.join(shlex.quote(a) for a in args[:20])}")
        if len(args) > 20:
            print(f"      ... ({len(args) - 20} more args)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


def trim_video(
    source: str,
    output: str,
    start: float | None = None,
    duration: float | None = None,
    crf: int = 18,
) -> None:
    """
    Trim a video to a time range (re-encode for frame accuracy).
    Resets timestamps so the output starts cleanly at PTS 0 — this prevents
    1-frame offset issues when multiple tools decode the same trimmed file.
    """
    args = []
    if start is not None:
        args += ["-ss", str(start)]
    if duration is not None:
        args += ["-t", str(duration)]
    args += ["-i", source]
    args += [
        "-vf",
        "setpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        output,
    ]
    run_ffmpeg(args, "Trimming to time range")


def scale_video(
    source: str,
    output: str,
    out_w: int,
    out_h: int,
    flags: str = "bicubic",
    crf: int = 18,
) -> None:
    """Scale a video to exact dimensions."""
    args = [
        "-i",
        source,
        "-vf",
        f"scale={out_w}:{out_h}:flags={flags}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        output,
    ]
    run_ffmpeg(args, f"Scale to {out_w}x{out_h} ({flags})")


def crop_video(
    source: str,
    output: str,
    crop_w: int,
    crop_h: int,
    crop_x: int,
    crop_y: int,
    crf: int = 18,
) -> None:
    """Crop a region from a video."""
    args = [
        "-i",
        source,
        "-vf",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        output,
    ]
    run_ffmpeg(args, f"Crop {crop_w}x{crop_h} from ({crop_x},{crop_y})")


def make_comparison(
    left: str,
    right: str,
    output: str,
    left_label: str = "Bicubic",
    right_label: str = "RTX Video SR",
    font_size: int = 28,
    font: str | None = None,
    font_sub: str | None = None,
    crf: int = 18,
) -> None:
    """Create a side-by-side video with text labels."""
    # Check frame counts — if they differ, trim both to the shorter one.
    # rtx_enhance.py can produce 1 fewer frame than ffmpeg's scaler due to
    # raw frame I/O boundary handling.
    left_info = probe_video(left)
    right_info = probe_video(right)

    def _count_frames(path: str) -> int:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-print_format",
                "default=nokey=1:noprint_wrappers=1",
                path,
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            return int(out)
        except ValueError:
            return 0

    left_frames = _count_frames(left)
    right_frames = _count_frames(right)

    frames_arg = []
    if left_frames > 0 and right_frames > 0 and left_frames != right_frames:
        min_frames = min(left_frames, right_frames)
        print(
            f"  Frame count mismatch: left={left_frames}, right={right_frames} "
            f"— trimming both to {min_frames}"
        )
        frames_arg = ["-frames:v", str(min_frames)]

    def _resolve_font(font_spec: str | None) -> str:
        """Resolve a font name or path to a drawtext fontfile= arg."""
        if not font_spec:
            return ""
        if os.path.isfile(font_spec):
            font_path = font_spec
        else:
            result = subprocess.run(
                ["fc-match", font_spec, "--format=%{file}"],
                capture_output=True,
                text=True,
            )
            font_path = result.stdout.strip() if result.returncode == 0 else ""
        if font_path:
            escaped = font_path.replace("\\", "\\\\").replace(":", "\\:")
            return f"fontfile='{escaped}':"
        return ""

    font_arg_title = _resolve_font(font)
    font_arg_sub = _resolve_font(font_sub) if font_sub else _resolve_font(font)

    def _drawtext_chain(label: str, font_sz: int, align: str = "left") -> str:
        """
        Build drawtext filter(s) for a label that may contain newlines.
        First line uses the title font (bold/italic), subsequent lines use
        the subtitle font (lighter weight) at a smaller size.
        """
        lines = label.split("\n")
        n = len(lines)
        margin_x = 64
        margin_y = 28
        sub_sz = int(font_sz * 0.7)
        filters = []

        # Calculate total height from bottom for positioning
        # Title line height + subtitle line heights
        total_h = font_sz  # title
        if n > 1:
            total_h += -6  # negative gap to tighten title/subtitle spacing
            total_h += (n - 1) * (sub_sz + 4)  # subtitle lines

        for i, line in enumerate(lines):
            escaped = (
                line.replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
            )

            is_title = i == 0
            cur_font = font_arg_title if is_title else font_arg_sub
            cur_sz = font_sz if is_title else sub_sz

            # Stack from bottom up
            if i == 0:
                # Title line — sits above all subtitle lines
                y_offset = margin_y + total_h
            else:
                # Subtitle lines — count from bottom
                lines_below = n - 1 - i
                y_offset = margin_y + sub_sz + lines_below * (sub_sz + 4)

            y_expr = f"h-{y_offset}"
            if align == "left":
                x_expr = str(margin_x)
            else:
                x_expr = f"w-tw-{margin_x}"

            filters.append(
                f"drawtext={cur_font}text='{escaped}':"
                f"fontsize={cur_sz}:fontcolor=white:"
                f"x={x_expr}:y={y_expr}"
            )
        return ",".join(filters)

    left_chain = _drawtext_chain(left_label, font_size, align="left")
    right_chain = _drawtext_chain(right_label, font_size, align="right")

    filter_complex = (
        f"[0:v]{left_chain}[left];"
        f"[1:v]{right_chain}[right];"
        f"[left][right]hstack=inputs=2:shortest=1[out]"
    )
    args = (
        [
            "-i",
            left,
            "-i",
            right,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
        ]
        + frames_arg
        + [output]
    )
    run_ffmpeg(args, "Side-by-side with labels")


def twitter_encode(source: str, output: str, crf: int = 18) -> None:
    """
    Ensure Twitter/social media compatibility: h264, yuv420p, faststart.
    No scaling — the comparison should already be at the target resolution.
    """
    info = probe_video(source)
    src_w, src_h = info["width"], info["height"]

    # Ensure even dimensions
    out_w = src_w + (src_w % 2)
    out_h = src_h + (src_h % 2)

    vf_parts = []
    if out_w != src_w or out_h != src_h:
        vf_parts.append(f"scale={out_w}:{out_h}")

    args = ["-i", source]
    if vf_parts:
        args += ["-vf", ",".join(vf_parts)]
    args += [
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output,
    ]
    run_ffmpeg(args, f"Twitter-compatible encode ({out_w}x{out_h}, h264)")


# ─── RTX SR upscale via rtx_enhance.py ─────────────────────────────────────


def run_rtx_enhance(
    source: str,
    output: str,
    scale: float = 2.0,
    rtx_quality: str = "ULTRA",
    decompress: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """Run rtx_enhance.py to produce the RTX SR version."""
    script = Path(__file__).parent / "rtx_enhance.py"
    cmd = [
        sys.executable,
        str(script),
        source,
        output,
        "--scale",
        str(scale),
        "--rtx-quality",
        rtx_quality,
        "--codec",
        "h264",
        "--crf",
        "18",
        "--preset",
        "medium",
        "--no-audio",
    ]
    if decompress:
        cmd.append("--decompress")
    if extra_args:
        cmd += extra_args

    print(f"  Running RTX SR {scale}x upscale ({rtx_quality})...")
    print(f"    $ {' '.join(shlex.quote(str(c)) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"Error: rtx_enhance.py failed (exit {result.returncode})", file=sys.stderr
        )
        sys.exit(1)


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparison: Bicubic vs RTX Video SR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The workflow:
  1. Upscale the full source frame with bicubic and RTX SR
  2. Crop the same region from both upscaled outputs
  3. hstack with labels -> comparison video

Crop coordinates are in the UPSCALED frame. For a 540p source at 2x:
  upscaled frame = 1920x1080
  --crop 960x1080+0+0    -> left half
  --crop 960x1080+480+0  -> center
  --crop 960x1080+960+0  -> right half

For a 1920x1080 output, each side should be 960x1080.

Examples:
  # 2x upscale, crop center 960x1080, get 1920x1080 comparison
  python compare.py source_540p.mp4 --crop 960x1080+480+0

  # 4x upscale from 480p source — dramatic quality difference
  python compare.py source_480p.mp4 --scale 4 --crop 960x1080+800+420

  # 6-second clip, Twitter-ready
  python compare.py source.mp4 --crop 960x1080+200+0 --start 5 --duration 6 --twitter

  # With existing RTX SR output
  python compare.py source.mp4 --rtxsr rtxsr.mp4 --crop 960x1080+480+0

  # Custom labels
  python compare.py source.mp4 --crop 960x1080+480+0 --right-label "RTX VSR + DeH264"
        """,
    )

    parser.add_argument("source", help="Source video file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output comparison video (default: <source>_compare.mp4)",
    )

    # Upscaling
    upscale = parser.add_argument_group("Upscaling")
    upscale.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Upscale factor for both bicubic and RTX SR (default: 2.0)",
    )
    upscale.add_argument(
        "--rtxsr",
        default=None,
        help="Path to existing RTX SR upscaled video (skip RTX upscaling step)",
    )
    upscale.add_argument(
        "--rtx-quality",
        default="ULTRA",
        help="RTX VSR quality level (default: ULTRA)",
    )
    upscale.add_argument(
        "--decompress",
        action="store_true",
        help="Enable DeH264 artifact removal in RTX SR pass",
    )

    # Clip & crop selection
    clip = parser.add_argument_group("Clip & crop selection")
    clip.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds",
    )
    clip.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (tip: keep under 10s for social media)",
    )
    clip.add_argument(
        "--crop",
        default=None,
        help="Crop region in UPSCALED frame: WxH+X+Y (e.g. 960x1080+480+0). "
        "Applied after upscaling so RTX SR gets full-frame context. "
        "For 1920x1080 output, use 960x1080.",
    )

    # Labels
    labels = parser.add_argument_group("Labels")
    labels.add_argument(
        "--left-label",
        default=None,
        help="Label for left side (default: 'Original\\n<resolution>')",
    )
    labels.add_argument(
        "--right-label",
        default=None,
        help="Label for right side (default: 'RTX Video SR\\n<resolution>')",
    )
    labels.add_argument(
        "--font-size",
        type=int,
        default=88,
        help="Label font size (default: 88)",
    )
    labels.add_argument(
        "--font",
        default=FONT_TITLE,
        help=f"Title font — path to .ttf/.otf (default: {FONT_TITLE})",
    )
    labels.add_argument(
        "--font-sub",
        default=FONT_SUB,
        help=f"Subtitle font — path to .ttf/.otf (default: {FONT_SUB})",
    )

    # Output options
    out = parser.add_argument_group("Output options")
    out.add_argument(
        "--twitter",
        action="store_true",
        help="Produce a Twitter-compatible version (h264, yuv420p, faststart)",
    )
    out.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for all encodes (default: 18)",
    )
    out.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files in <source>_compare_files/",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    if args.rtxsr and not os.path.isfile(args.rtxsr):
        print(f"Error: RTX SR file not found: {args.rtxsr}", file=sys.stderr)
        sys.exit(1)

    # ── Probe source ────────────────────────────────────────────────────
    print(f"Probing source: {args.source}")
    src_info = probe_video(args.source)
    src_w, src_h = src_info["width"], src_info["height"]
    print(
        f"  Source: {src_w}x{src_h} @ {src_info['fps']:.1f}fps, "
        f"{src_info['duration']:.1f}s"
    )

    # Compute upscaled dimensions
    up_w = int(src_w * args.scale)
    up_h = int(src_h * args.scale)
    # RTX requires dimensions divisible by 8
    up_w = max(8, (up_w // 8) * 8)
    up_h = max(8, (up_h // 8) * 8)
    print(f"  Upscaled: {up_w}x{up_h} ({args.scale}x)")

    # Auto-generate labels with resolution info
    if args.left_label is None:
        args.left_label = f"ORIGINAL\n{src_w}x{src_h}"
    if args.right_label is None:
        args.right_label = f"RTX VSR\n{up_w}x{up_h}"

    # Parse crop (in upscaled frame coordinates)
    if args.crop:
        crop_w, crop_h, crop_x, crop_y = parse_crop(args.crop)
        # Validate crop fits within upscaled frame
        if crop_x + crop_w > up_w or crop_y + crop_h > up_h:
            print(
                f"Error: crop {crop_w}x{crop_h}+{crop_x}+{crop_y} exceeds "
                f"upscaled frame {up_w}x{up_h}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Crop: {crop_w}x{crop_h} from ({crop_x},{crop_y}) in upscaled frame")
        print(f"  Output: {crop_w * 2}x{crop_h} (hstack)")
    else:
        crop_w = crop_h = crop_x = crop_y = 0
        print(f"  Output: {up_w * 2}x{up_h} (hstack, no crop)")

    # Derive output filename
    source_stem = Path(args.source).stem
    source_dir = Path(args.source).parent
    if args.output:
        output_path = args.output
    else:
        suffix = "_twitter" if args.twitter else ""
        output_path = str(source_dir / f"{source_stem}_compare{suffix}.mp4")

    # Track intermediate files for cleanup
    intermediates: list[str] = []
    has_trim = args.start is not None or args.duration is not None

    try:
        with tempfile.TemporaryDirectory(prefix="rtx_compare_") as tmpdir:
            # ── Step 1: Trim source if needed ───────────────────────────

            if has_trim:
                trimmed_source = os.path.join(tmpdir, "source_trimmed.mp4")
                print(f"\n[1/4] Trimming source...")
                trim_video(
                    args.source,
                    trimmed_source,
                    start=args.start,
                    duration=args.duration,
                    crf=args.crf,
                )
                source_path = trimmed_source
                intermediates.append(trimmed_source)
            else:
                source_path = args.source
                print(f"\n[1/4] No trim, using full source")

            # ── Step 2: Bicubic upscale (full frame) ────────────────────

            bicubic_full = os.path.join(tmpdir, "bicubic_full.mp4")
            print(f"\n[2/4] Bicubic {args.scale}x upscale (full frame)...")
            scale_video(
                source_path,
                bicubic_full,
                up_w,
                up_h,
                flags="bicubic",
                crf=args.crf,
            )
            intermediates.append(bicubic_full)

            # ── Step 3: RTX SR upscale (full frame) ─────────────────────

            if args.rtxsr:
                if has_trim:
                    # Trim the existing RTX SR file to match
                    rtxsr_full = os.path.join(tmpdir, "rtxsr_trimmed.mp4")
                    print(f"\n[3/4] Trimming existing RTX SR video...")
                    trim_video(
                        args.rtxsr,
                        rtxsr_full,
                        start=args.start,
                        duration=args.duration,
                        crf=args.crf,
                    )
                    intermediates.append(rtxsr_full)
                else:
                    rtxsr_full = args.rtxsr
                    print(f"\n[3/4] Using existing RTX SR video: {args.rtxsr}")
            else:
                rtxsr_full = os.path.join(tmpdir, "rtxsr_full.mp4")
                print(f"\n[3/4] RTX SR {args.scale}x upscale (full frame)...")
                run_rtx_enhance(
                    source_path,
                    rtxsr_full,
                    scale=args.scale,
                    rtx_quality=args.rtx_quality,
                    decompress=args.decompress,
                )
                intermediates.append(rtxsr_full)

            # ── Crop both to the same region ────────────────────────────

            if args.crop:
                bicubic_side = os.path.join(tmpdir, "bicubic_crop.mp4")
                rtxsr_side = os.path.join(tmpdir, "rtxsr_crop.mp4")

                print(f"\n  Cropping both to {crop_w}x{crop_h}...")

                # Crop bicubic
                crop_video(
                    bicubic_full,
                    bicubic_side,
                    crop_w,
                    crop_h,
                    crop_x,
                    crop_y,
                    crf=args.crf,
                )

                # For RTX SR, map crop coords if its dimensions differ
                # (e.g. user provided --rtxsr with different scale)
                rtx_info = probe_video(rtxsr_full)
                if rtx_info["width"] != up_w or rtx_info["height"] != up_h:
                    rx = rtx_info["width"] / up_w
                    ry = rtx_info["height"] / up_h
                    rtx_cx = int(crop_x * rx)
                    rtx_cy = int(crop_y * ry)
                    rtx_cw = int(crop_w * rx)
                    rtx_ch = int(crop_h * ry)
                    print(
                        f"  RTX SR is {rtx_info['width']}x{rtx_info['height']} "
                        f"(not {up_w}x{up_h}), adjusting crop to "
                        f"{rtx_cw}x{rtx_ch}+{rtx_cx}+{rtx_cy}"
                    )
                    # Crop then scale to match bicubic crop size
                    run_ffmpeg(
                        [
                            "-i",
                            rtxsr_full,
                            "-vf",
                            f"crop={rtx_cw}:{rtx_ch}:{rtx_cx}:{rtx_cy},"
                            f"scale={crop_w}:{crop_h}:flags=lanczos",
                            "-c:v",
                            "libx264",
                            "-crf",
                            str(args.crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-an",
                            rtxsr_side,
                        ],
                        f"Crop + scale RTX SR to {crop_w}x{crop_h}",
                    )
                else:
                    crop_video(
                        rtxsr_full,
                        rtxsr_side,
                        crop_w,
                        crop_h,
                        crop_x,
                        crop_y,
                        crf=args.crf,
                    )

                intermediates += [bicubic_side, rtxsr_side]
            else:
                # No crop — use full upscaled frames
                bicubic_side = bicubic_full
                rtxsr_side = rtxsr_full

                # If dimensions don't match, scale bicubic to match RTX SR
                rtx_info = probe_video(rtxsr_side)
                bic_info = probe_video(bicubic_side)
                if (
                    bic_info["width"] != rtx_info["width"]
                    or bic_info["height"] != rtx_info["height"]
                ):
                    print(
                        f"  Dimension mismatch: bicubic {bic_info['width']}x"
                        f"{bic_info['height']} vs RTX SR {rtx_info['width']}x"
                        f"{rtx_info['height']} — rescaling bicubic to match"
                    )
                    bicubic_matched = os.path.join(tmpdir, "bicubic_matched.mp4")
                    scale_video(
                        bicubic_side,
                        bicubic_matched,
                        rtx_info["width"],
                        rtx_info["height"],
                        flags="bicubic",
                        crf=args.crf,
                    )
                    bicubic_side = bicubic_matched
                    intermediates.append(bicubic_matched)

            # ── Step 4: Side-by-side comparison ─────────────────────────

            if args.twitter:
                comparison_path = os.path.join(tmpdir, "comparison.mp4")
            else:
                comparison_path = output_path

            print(f"\n[4/4] Creating side-by-side comparison...")
            make_comparison(
                bicubic_side,
                rtxsr_side,
                comparison_path,
                left_label=args.left_label,
                right_label=args.right_label,
                font_size=args.font_size,
                font=args.font,
                font_sub=args.font_sub,
                crf=args.crf,
            )

            # ── Optional Twitter encode ─────────────────────────────────

            if args.twitter:
                print(f"\n[bonus] Twitter-compatible encode...")
                twitter_encode(comparison_path, output_path, crf=args.crf)

            # ── Keep intermediates if requested ─────────────────────────

            if args.keep_intermediates:
                keep_dir = source_dir / f"{source_stem}_compare_files"
                keep_dir.mkdir(exist_ok=True)
                import shutil

                for f in intermediates:
                    if os.path.isfile(f):
                        dest = keep_dir / Path(f).name
                        shutil.copy2(f, dest)
                        print(f"  Kept: {dest}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────
    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        out_info = probe_video(output_path)
        print(f"\nDone!")
        print(f"  Output: {output_path}")
        print(f"  Size:   {size_mb:.1f} MB")
        print(f"  Dims:   {out_info['width']}x{out_info['height']}")
        print(f"  Length: {out_info['duration']:.1f}s")
    else:
        print(f"\nError: output file not created", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
