"""
VapourSynth filter for native mpv — realtime DeH264 cleanup only.

Runs the lightweight SuperUltraCompact safetensors model in-process and keeps the
video at the original resolution. No RTX VSR/nvvfx upscaling is used.

Usage:
    mpv video.mp4 --vf=vapoursynth=deh264_direct.py:concurrent-frames=1

Requires native mpv built against VapourSynth, and the Python environment must
have torch + spandrel installed.
"""

import ctypes
import os
import sys
import threading
import time

import numpy as np
import torch
import vapoursynth as vs

core = vs.core

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "cli",
        "models",
        "deH264_SuperUltraCompact.safetensors",
    )
)

_gpu_lock = threading.Lock()
_model = None
_frame_count = 0
_t_total = 0.0


def _load_model() -> torch.nn.Module:
    """Load the SuperUltraCompact DeH264 model."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"DeH264 model not found: {MODEL_PATH}")

    import spandrel

    model_desc = spandrel.ModelLoader(device="cuda").load_from_file(MODEL_PATH)
    model_desc.model.eval()
    print(f"DeH264: loaded {MODEL_PATH}", file=sys.stderr)
    return model_desc.model


def _copy_plane_to_frame(fout: vs.VideoFrame, plane: int, data: np.ndarray) -> None:
    """Copy one contiguous uint8 plane into a VapourSynth output frame."""
    height, width = data.shape
    ptr = ctypes.cast(fout.get_write_ptr(plane), ctypes.c_void_p).value
    stride = fout.get_stride(plane)
    plane_data = data.tobytes()

    if stride == width:
        ctypes.memmove(ptr, plane_data, width * height)
        return

    for row in range(height):
        ctypes.memmove(
            ptr + row * stride,
            plane_data[row * width : (row + 1) * width],
            width,
        )


@torch.inference_mode()
def make_frame(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
    """Process one RGB frame with the DeH264 model."""
    global _frame_count, _t_total

    t_start = time.time()
    src = f[0]
    height = src.height
    width = src.width

    frame = np.empty((3, height, width), dtype=np.uint8)
    frame[0] = np.frombuffer(bytes(src[0]), dtype=np.uint8).reshape(height, width)
    frame[1] = np.frombuffer(bytes(src[1]), dtype=np.uint8).reshape(height, width)
    frame[2] = np.frombuffer(bytes(src[2]), dtype=np.uint8).reshape(height, width)

    with _gpu_lock:
        tensor = (
            torch.from_numpy(frame)
            .cuda(non_blocking=True)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .contiguous()
        )
        output = _model(tensor).squeeze(0).clamp_(0, 1)
        result = output.mul_(255.0).byte().contiguous().cpu().numpy()

    fout = f[1].copy()
    for plane in range(3):
        _copy_plane_to_frame(fout, plane, result[plane])

    fout.props.update(src.props)

    _frame_count += 1
    _t_total += time.time() - t_start
    if _frame_count % 100 == 0:
        fps = _frame_count / _t_total if _t_total > 0 else 0.0
        print(f"DeH264: {_frame_count} frames, {fps:.1f} fps", file=sys.stderr)

    return fout


# ─── Filter setup ──────────────────────────────────────────────────────────

clip = video_in  # noqa: F821
clip_rgb = core.resize.Bilinear(clip, format=vs.RGB24, matrix_in_s="709")

_model = _load_model()

blank = core.std.BlankClip(clip_rgb, format=vs.RGB24)
clip_out = core.std.ModifyFrame(blank, [clip_rgb, blank], make_frame)
clip_out = core.resize.Bilinear(clip_out, format=vs.YUV420P8, matrix_s="709")
clip_out.set_output()
