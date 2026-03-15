# rtx-upscale

Video and image upscaling/enhancement using Nvidia's [`nvidia-vfx`](https://pypi.org/project/nvidia-vfx/) SDK (RTX Video Super Resolution). The SDK supports AI upscaling (2x/3x/4x), denoising, and deblurring on RTX GPUs.

Three independent tools — install only what you need:

- **`img/`** — Image upscaling, denoising, and deblurring (single files or batch)
- **`cli/`** — Batch video processing (ffmpeg decode → optional DeH264 artifact removal → RTX upscale → ffmpeg encode)
- **`mpv/`** — Real-time video upscaling during mpv playback via VapourSynth + shared memory

## Image Processing — `img/`

Upscale, denoise, and deblur images. Supports chaining multiple passes (e.g. denoise → deblur → upscale).

### Requirements

- Nvidia RTX GPU (20-series or newer)
- Python 3.10+

### Install

```bash
pip install -r img/requirements.txt
```

### Usage

```bash
# 2x upscale
python img/rtx_image.py photo.jpg upscaled.png

# 4x upscale with high-bitrate mode (for clean sources)
python img/rtx_image.py photo.png big.png --scale 4 --quality HIGHBITRATE_ULTRA

# Denoise then upscale
python img/rtx_image.py noisy.jpg clean_big.png --denoise

# Denoise + deblur then upscale (max cleanup)
python img/rtx_image.py rough.jpg polished.png --denoise --deblur

# Denoise only (no upscale)
python img/rtx_image.py noisy.jpg clean.png --denoise --no-upscale

# Batch process a directory
python img/rtx_image.py input_dir/ output_dir/ --scale 2 --denoise
```

### Options

```
Upscaling:
  --scale FACTOR            Upscale factor (default: 2.0)
  --quality LEVEL           RTX quality: LOW/MEDIUM/HIGH/ULTRA, HIGHBITRATE_* (default: ULTRA)
  --no-upscale              Skip upscaling (only run denoise/deblur passes)

Pre-processing (applied before upscaling):
  --denoise                 Run a denoise pass
  --denoise-strength LEVEL  LOW/MEDIUM/HIGH/ULTRA (default: ULTRA)
  --deblur                  Run a deblur pass
  --deblur-strength LEVEL   LOW/MEDIUM/HIGH/ULTRA (default: ULTRA)

Output:
  --format FMT              Output format (png, jpg, webp). Auto-detected from extension
```

## Batch CLI — `cli/`

Upscale video files with a threaded pipeline: ffmpeg decode &rarr; GPU processing &rarr; ffmpeg encode.

### Requirements

- Nvidia RTX GPU (20-series or newer)
- Python 3.10+
- ffmpeg in PATH

### Install

```bash
pip install -r cli/requirements.txt
```

(`spandrel` is only needed for the `ultrafast` or `best` DeH264 presets. The default `fast` preset uses the bundled RTMoSR architecture.)

### Basic usage

```bash
# 2x upscale with RTX VSR
python cli/rtx_enhance.py input.mp4 output.mkv

# With DeH264 artifact removal before upscaling
python cli/rtx_enhance.py input.mp4 output.mkv --decompress

# Decompress only (no upscaling) — clean up a low-bitrate source
python cli/rtx_enhance.py input.mp4 output.mkv --decompress --no-upscale
```

### Options

```
Pipeline stages:
  --decompress              Enable DeH264 artifact removal before upscaling
  --no-upscale              Skip RTX upscaling (decompress only, requires --decompress)

DeH264 artifact removal:
  --deh264-preset {fast,ultrafast,best}
                            Quality/speed preset (default: fast)
      ultrafast — SuperUltraCompact, 43K params, fastest
      fast      — RTMoSR, 2.8M params, good all-rounder
      best      — RealPLKSR, 7.4M params, strongest artifact removal
  --deh264-model PATH       Custom model file (overrides --deh264-preset)
  --tile-size N             Process in tiles of NxN (0 = full frame, use if OOM)

RTX upscaling:
  --scale FACTOR            Upscale factor (default: 2.0)
  --target-width W          Target output width (overrides --scale)
  --target-height H         Target output height (overrides --scale)
  --rtx-quality {LOW,MEDIUM,HIGH,ULTRA}
                            RTX VSR quality (default: ULTRA)

Output encoding:
  --codec CODEC             av1, av1_nvenc, hevc_nvenc, nvenc, h264, h265, prores, etc. (default: av1)
  --crf N                   CRF/CQ quality (default: 25)
  --preset N                Encoder preset (default: 4)
  --no-audio                Don't copy audio from input
  --ffmpeg-args ...         Extra ffmpeg output arguments
```

### Examples

```bash
# Maximum quality: best decompression + 4x upscale, HEVC output
python cli/rtx_enhance.py input.mp4 output.mkv --decompress --deh264-preset best --scale 4 --codec hevc

# Fastest possible processing
python cli/rtx_enhance.py input.mp4 output.mkv --deh264-preset ultrafast --rtx-quality LOW

# Target exact resolution
python cli/rtx_enhance.py input.mp4 output_4k.mkv --target-width 3840 --target-height 2160

# SVT-AV1 software encoding (better compression, slower)
python cli/rtx_enhance.py input.mp4 output.mkv --codec av1 --crf 25 --preset 4

# ProRes for editing
python cli/rtx_enhance.py input.mp4 output.mov --codec prores
```

### DeH264 models

Three artifact removal models are included for cleaning up compression artifacts before upscaling:

| Preset | Model | Size | Params | Speed | Best for |
|--------|-------|------|--------|-------|----------|
| `ultrafast` | SuperUltraCompact | 172K | 43K | Fastest | Light cleanup on decent sources |
| `fast` | RTMoSR | 11M | 2.8M | Fast | Good all-rounder (default) |
| `best` | RealPLKSR | 29M | 7.4M | Slowest | Heavy artifact removal on low-bitrate footage |

Models from [Phhofm/models](https://github.com/Phhofm/models) and [TNTwise/real-video-enhancer-models](https://github.com/TNTwise/real-video-enhancer-models).

RTMoSR architecture from [rewaifu/RTMoSR](https://github.com/rewaifu/RTMoSR) (MIT license).

## Real-time mpv playback — `mpv/`

Watch any video with RTX Super Resolution applied in real-time. Uses a server/client architecture with file-backed mmap for zero-copy frame transfer.

### Requirements

- Nvidia RTX GPU (20-series or newer)
- Python 3.10+
- [VapourSynth](https://www.vapoursynth.com/doc/installation.html) (`pip install vapoursynth`)
- [mpv](https://mpv.io/) with VapourSynth support (built with `--enable-vapoursynth`)

### Install

```bash
pip install -r mpv/requirements.txt
pip install vapoursynth
```

### Setup

**Terminal 1** — Start the RTX VSR server:

```bash
python mpv/rtx_vsr_server.py
```

**Terminal 2** — Play a video with the filter:

```bash
mpv video.mp4 --hwdec=auto-copy --vf="vapoursynth=file=$(pwd)/mpv/rtx_vsr.py:concurrent-frames=1"
```

Or use the direct in-process filter (no server needed, faster):

```bash
bash mpv/mpv-rtx video.mp4
```

The direct filter runs nvvfx in-process and avoids the mmap/socket overhead, so it comfortably handles 24fps+ content. The server approach struggles to maintain 24fps due to IPC latency. **Use the direct filter if your mpv is native (not flatpak).**

### Server options

```bash
python mpv/rtx_vsr_server.py --quality ULTRA    # default
python mpv/rtx_vsr_server.py --quality MEDIUM    # faster
```

### How it works

**Direct filter** (recommended) — nvvfx runs inside the VapourSynth filter process. Simple, fast, no IPC overhead.

```
mpv (native)
+---------------------------+
| decode video              |
| VapourSynth filter        |
|   torch + nvvfx in-process|
|   RTX VSR upscale         |
| display upscaled frame    |
+---------------------------+
```

**Server/client** (for flatpak mpv) — a separate server process handles GPU work, communicating via mmap + Unix socket. This exists because flatpak mpv can't load native CUDA libraries directly.

```
mpv (flatpak)                              Host Python
+--------------------+                   +--------------------+
| decode video       |                   | rtx_vsr_server.py  |
| VapourSynth filter |                   |   torch + nvvfx    |
|   write to mmap    |--signal socket--> |   read from mmap   |
|   read from mmap   |<-signal socket--  |   RTX VSR upscale  |
| display upscaled   |                   |   write to mmap    |
+--------------------+                   +--------------------+
```

The IPC overhead makes the server approach slower — it struggles to maintain 24fps. Use the direct filter whenever possible.

### Tips

- `--hwdec=auto-copy` lets the GPU decode video (NVDEC) while keeping frames accessible to VapourSynth
- For the server approach: `concurrent-frames=1` is required, and the server must be running before you start mpv
- Seeking works but may briefly stutter as the connection recovers

## License

RTMoSR architecture code is MIT licensed. The nvidia-vfx SDK is proprietary Nvidia software. DeH264 model weights are from their respective authors.
