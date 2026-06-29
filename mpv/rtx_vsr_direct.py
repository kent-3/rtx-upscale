"""
VapourSynth filter for native mpv — direct RTX VSR, no server needed.
Loads nvvfx and torch in-process for zero-overhead GPU upscaling.
Optionally runs DeH264 artifact removal before upscaling.

Usage:
    ~/ai/mpv-native/bin/mpv video.mp4 --vf=vapoursynth=rtx_vsr_direct.py:concurrent-frames=1

Config (~/.config/rtx-vsr.conf):
    quality=ULTRA
    decompress=true
    scale=1

Runtime override:
    RTX_VSR_SCALE=1 bash mpv/mpv-rtx video.mp4

Requires native (non-flatpak) mpv built against VapourSynth, and
the Python environment must have torch + nvvfx installed.
DeH264 requires spandrel: pip install spandrel
"""

import ctypes
import os
import sys
import threading

import numpy as np
import torch
import nvvfx
import vapoursynth as vs
import time

_gpu_lock = threading.Lock()

TARGET_W = 3840
TARGET_H = 2160
MAX_SCALE = 2.0

core = vs.core

# ─── DeH264 setup ──────────────────────────────────────────────────────────

_deh264_model = None


def _init_deh264():
    """Load the DeH264 SuperUltraCompact model for artifact removal."""
    global _deh264_model
    import spandrel

    # Find model relative to this script's sibling cli/models/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, "..", "cli", "models", "deH264_SuperUltraCompact.safetensors"
    )

    if not os.path.isfile(model_path):
        print(f"DeH264: model not found at {model_path}, skipping", file=sys.stderr)
        return

    model_desc = spandrel.ModelLoader(device="cuda").load_from_file(model_path)
    model_desc.model.eval()
    _deh264_model = model_desc.model
    print(f"DeH264: loaded SuperUltraCompact (artifact removal)", file=sys.stderr)


# ─── RTX VSR setup ─────────────────────────────────────────────────────────

_sr = None
_out_w = 0
_out_h = 0


def _init_upscaler(in_w, in_h, out_w, out_h, quality="ULTRA"):
    """Initialize the RTX VSR upscaler."""
    global _sr, _out_w, _out_h

    ql = nvvfx.effects.QualityLevel
    quality_map = {
        name: getattr(ql, name)
        for name in dir(ql)
        if not name.startswith("_") and isinstance(getattr(ql, name), int)
    }
    q = quality_map.get(quality, ql.ULTRA)

    _sr = nvvfx.VideoSuperRes(q)
    _sr.output_width = out_w
    _sr.output_height = out_h
    _sr.load()
    _out_w = out_w
    _out_h = out_h
    print(f"RTX VSR: {in_w}x{in_h} -> {out_w}x{out_h} ({quality})", file=sys.stderr)


# ─── Config ────────────────────────────────────────────────────────────────

_quality = "ULTRA"
_decompress = False
_forced_scale = None
_config_path = os.path.expanduser("~/.config/rtx-vsr.conf")
if os.path.isfile(_config_path):
    with open(_config_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("quality="):
                _quality = _line.split("=", 1)[1].strip()
            elif _line.startswith("decompress="):
                _val = _line.split("=", 1)[1].strip().lower()
                _decompress = _val in ("true", "1", "yes")
            elif _line.startswith("scale="):
                _forced_scale = float(_line.split("=", 1)[1].strip())

if os.environ.get("RTX_VSR_SCALE"):
    _forced_scale = float(os.environ["RTX_VSR_SCALE"])

_frame_count = 0
_t_total = 0


@torch.inference_mode()
def make_frame(n, f):
    """Process a single frame through DeH264 + RTX VSR — all on GPU."""
    global _frame_count, _t_total
    t_start = time.time()

    src = f[0]
    width = src.width
    height = src.height

    # Read planar RGB from VS frame into a contiguous numpy array
    t0 = time.time()
    r = bytes(src[0])
    g = bytes(src[1])
    b = bytes(src[2])
    frame = np.empty((3, height, width), dtype=np.uint8)
    frame[0] = np.frombuffer(r, dtype=np.uint8).reshape(height, width)
    frame[1] = np.frombuffer(g, dtype=np.uint8).reshape(height, width)
    frame[2] = np.frombuffer(b, dtype=np.uint8).reshape(height, width)
    t_read = time.time() - t0

    # Upload to GPU, run DeH264 + RTX VSR (locked for thread safety)
    t0 = time.time()
    with _gpu_lock:
        tensor = (
            torch.from_numpy(frame)
            .cuda(non_blocking=True)
            .float()
            .div_(255.0)
            .contiguous()
        )

        # DeH264 artifact removal (same resolution, cleans up compression)
        if _deh264_model is not None:
            tensor = _deh264_model(tensor.unsqueeze(0))
            tensor = tensor.squeeze(0).clamp_(0, 1).contiguous()

        # RTX VSR upscale
        dlpack_out = _sr.run(tensor).image
        output = torch.from_dlpack(dlpack_out)
        result = output.clamp_(0, 1).mul_(255.0).byte().contiguous().cpu().numpy()
    t_gpu = time.time() - t0

    # Write into output VS frame
    t0 = time.time()
    fout = f[1].copy()
    for p in range(3):
        ptr = ctypes.cast(fout.get_write_ptr(p), ctypes.c_void_p).value
        stride = fout.get_stride(p)
        plane_data = result[p].tobytes()
        if stride == _out_w:
            ctypes.memmove(ptr, plane_data, _out_w * _out_h)
        else:
            for row in range(_out_h):
                ctypes.memmove(
                    ptr + row * stride,
                    plane_data[row * _out_w : (row + 1) * _out_w],
                    _out_w,
                )
    t_write = time.time() - t0

    fout.props.update(src.props)

    _frame_count += 1
    _t_total += time.time() - t_start
    if _frame_count % 50 == 0:
        fps = _frame_count / _t_total
        deh_tag = "+DeH264 " if _deh264_model else ""
        print(
            f"RTX VSR: {deh_tag}{_frame_count} frames, {fps:.1f} fps | "
            f"read={t_read * 1000:.1f}ms gpu={t_gpu * 1000:.1f}ms "
            f"write={t_write * 1000:.1f}ms",
            file=sys.stderr,
        )

    return fout


# ─── Filter setup ──────────────────────────────────────────────────────────

clip = video_in  # noqa: F821

# Calculate scale to fit within the target box while respecting MAX_SCALE,
# unless a same-size or custom scale was explicitly requested.
if _forced_scale is not None:
    scale = _forced_scale
else:
    scale_w = TARGET_W / clip.width
    scale_h = TARGET_H / clip.height
    scale = min(scale_w, scale_h, 4.0, MAX_SCALE)

if scale <= 1.0 and _forced_scale is None:
    print(
        f"RTX VSR: {clip.width}x{clip.height} — skipping, already at target",
        file=sys.stderr,
    )
    clip.set_output()
else:
    clip_rgb = core.resize.Bilinear(clip, format=vs.RGB24, matrix_in_s="709")

    out_w = max(8, int(clip.width * scale) // 8 * 8)
    out_h = max(8, int(clip.height * scale) // 8 * 8)

    # Initialize DeH264 if enabled
    if _decompress:
        _init_deh264()

    # Initialize the upscaler
    _init_upscaler(clip.width, clip.height, out_w, out_h, _quality)

    blank = core.std.BlankClip(clip_rgb, width=out_w, height=out_h, format=vs.RGB24)
    clip_out = core.std.ModifyFrame(blank, [clip_rgb, blank], make_frame)
    clip_out = core.resize.Bilinear(clip_out, format=vs.YUV420P8, matrix_s="709")

    clip_out.set_output()
