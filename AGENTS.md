# AGENTS.md — rtx-upscale-cli

## Project Overview

Video upscaling and enhancement CLI using Nvidia RTX Video Super Resolution (`nvidia-vfx` SDK) and DeH264 artifact removal models. Two main components:

- **`rtx_enhance.py`** — Batch video processing pipeline (ffmpeg decode → optional DeH264 → RTX upscale → ffmpeg encode)
- **`mpv/`** — Real-time upscaling during mpv playback via VapourSynth filters

Pure Python project (no package manager, no build system). Dependencies installed via pip into a venv.

## Prerequisites

- **Nvidia RTX GPU** (20-series+)
- **Python 3.10+** with venv
- **ffmpeg** in PATH
- **CUDA** toolkit + `nvidia-vfx` SDK
- **mpv** with [VapourSynth](https://www.vapoursynth.com/doc/installation.html) support (for real-time playback).
  mpv must be built with `--enable-vapoursynth` (or `--enable-vapoursynth-lazy`).
  See [mpv's VapourSynth filter docs](https://mpv.io/manual/master/#video-filters-vapoursynth).

## Commands

### Install dependencies

```bash
pip install nvidia-vfx torch numpy spandrel
```

### Run the CLI

```bash
# Basic 2x upscale
python rtx_enhance.py input.mp4 output.mkv

# With DeH264 decompression
python rtx_enhance.py input.mp4 output.mkv --decompress

# Decompress only (no upscaling)
python rtx_enhance.py input.mp4 output.mkv --decompress --no-upscale
```

### Run the mpv real-time server

```bash
# Start server
python mpv/rtx_vsr_server.py

# Direct in-process filter (no server)
bash mpv/mpv-rtx video.mp4
```

### Lint / format

No linter or formatter is configured. Follow these conventions:

```bash
# If adding tooling, use:
ruff check .
ruff format .
```

### Tests

No test suite exists. The project is a CLI tool — test manually by running on sample video files.

## Architecture

```
rtx_enhance.py          Main CLI — threaded pipeline (decoder → GPU → encoder)
rtmosr.py               RTMoSR neural network architecture (MIT, from rewaifu/RTMoSR)
models/                 Pre-trained DeH264 model weights (.pth, .safetensors)
mpv/
  rtx_vsr_server.py     Standalone RTX VSR server (mmap + Unix socket IPC)
  rtx_vsr.py            VapourSynth filter client (connects to server via mmap)
  rtx_vsr_direct.py     VapourSynth filter with in-process nvvfx (no server)
  mpv-rtx               Shell wrapper to launch native mpv with direct filter
```

### Pipeline threading model (`rtx_enhance.py`)

Three threads connected by queues (depth 32):
1. **Decoder thread** — reads raw frames from ffmpeg, uploads to GPU via pinned memory
2. **GPU thread** — runs DeH264 model and/or RTX VSR inference
3. **Encoder thread** — writes processed frames to ffmpeg encoder

Main thread handles progress display. `threading.Event` for error propagation.

### Frame data format

- Internal format: `[3, H, W]` float32 tensor on CUDA, range [0, 1]
- I/O format: `HxWx3` uint8 RGB numpy array (ffmpeg rawvideo)
- mpv/VapourSynth uses planar RGB: `[3, H, W]` uint8 via mmap

## Code Style

### Python version

Target Python 3.10+. Use modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`.

### Imports

Standard library first, then third-party, then local. Sorted alphabetically within groups. Example from `rtx_enhance.py`:

```python
import argparse
import json
import os
import subprocess
import sys
import threading

import numpy as np
import torch

from rtmosr import RTMoSR
```

Conditional imports are acceptable for optional heavy dependencies (`nvvfx`, `spandrel`, `vapoursynth`) — import at point of use.

### Formatting

- 4-space indentation
- Double quotes for strings
- Line length ~88-100 chars (black-compatible)
- Trailing commas in multi-line collections
- Section separators: `# ─── Section Name ───...` (box-drawing chars)

### Type hints

Use type hints on function signatures. Not required on locals. Examples from codebase:

```python
def load_deh264_model(
    model_path: str, loader: str = "auto", device: str = "cuda"
) -> torch.nn.Module:

def deh264_process_frame(
    model: torch.nn.Module, frame: np.ndarray, tile_size: int = 0, device: str = "cuda"
) -> np.ndarray:
```

### Naming conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase` (e.g., `RTMoSR`, `CSELayer`, `GatedCNNBlock`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `QUEUE_DEPTH`, `SENTINEL`, `DEH264_PRESETS`)
- Private helpers: prefix with `_` (e.g., `_tiled_inference`, `_get_codec_args`, `_fmt_duration`)
- Module-level script globals: `_underscore_prefix` (e.g., `_sr`, `_conn`, `_gpu_lock`)

### Docstrings

Triple-quoted docstrings on public functions. Describe what, not how. Include parameter format when non-obvious:

```python
def deh264_process_frame(...) -> np.ndarray:
    """
    Run DeH264-RTMoSR on a single frame.
    frame: HxWx3 uint8 RGB numpy array
    Returns: HxWx3 uint8 RGB numpy array
    """
```

Module-level docstrings describe purpose, usage, and requirements.

### Error handling

- Use `sys.exit(1)` with a printed error for user-facing CLI errors
- Use `RuntimeError` for programmatic errors (e.g., "No video stream found")
- In threaded code, set `error_event` and print to stderr — don't raise
- Catch specific exceptions (`BrokenPipeError`, `ConnectionResetError`), not bare `except`

### Tensor operations

- Use `@torch.inference_mode()` on all inference functions (not `torch.no_grad()`)
- Use in-place ops where possible: `.div_(255.0)`, `.clamp_(0, 1)`, `.mul_(255.0)`
- Use `non_blocking=True` for CPU→GPU transfers
- Use pinned memory (`pin_memory=True`) for frame upload buffers
- Always `.contiguous()` before passing to nvvfx

### GPU / CUDA patterns

- Default device is `"cuda"` — this is a single-GPU tool
- RTX VSR dimensions must be divisible by 8: `max(8, (dim // 8) * 8)`
- nvvfx expects `[3, H, W]` float32 CUDA tensor; returns DLPack
- Use `torch.from_dlpack()` to consume nvvfx output (zero-copy)

### ffmpeg integration

- Use `subprocess.Popen` with pipes for frame I/O (stdin/stdout)
- Always use `-hide_banner -loglevel error` for clean output
- Use `ffprobe -print_format json` for metadata
- Buffer size: `width * height * 3 * N` frames for read-ahead

## Files You Should Not Modify

- `rtmosr.py` — vendored third-party architecture (MIT license, from rewaifu/RTMoSR)
- `models/*.pth`, `models/*.safetensors` — pre-trained weights (binary files)
