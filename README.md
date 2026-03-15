# rtx-upscale

Video upscaling and enhancement using Nvidia's [`nvidia-vfx`](https://pypi.org/project/nvidia-vfx/) SDK (RTX Video Super Resolution). The SDK supports AI upscaling (2x/3x/4x), denoising, and deblurring on RTX GPUs.

Two independent tools — install only what you need:

- **`cli/`** — Batch video processing (ffmpeg decode → optional DeH264 artifact removal → RTX upscale → ffmpeg encode)
- **`mpv/`** — Real-time upscaling during mpv playback via VapourSynth + shared memory

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

Or use the direct in-process filter (no server needed):

```bash
bash mpv/mpv-rtx video.mp4
```

### Server options

```bash
python mpv/rtx_vsr_server.py --quality ULTRA    # default
python mpv/rtx_vsr_server.py --quality MEDIUM    # faster
```

### How it works

```
mpv (flatpak or native)                    Host Python
+--------------------+                   +--------------------+
| decode video       |                   | rtx_vsr_server.py  |
| VapourSynth filter |                   |   torch + nvvfx    |
|   write to mmap    |--signal socket--> |   read from mmap   |
|   read from mmap   |<-signal socket--  |   RTX VSR upscale  |
| display 4K         |                   |   write to mmap    |
+--------------------+                   +--------------------+
```

Frame data is shared via memory-mapped files in `/tmp` (zero-copy). Only tiny 12-byte control messages go through the Unix socket. This architecture works across flatpak sandboxes.

### Tips

- `--hwdec=auto-copy` lets the GPU decode video (NVDEC) while keeping frames accessible to VapourSynth
- `concurrent-frames=1` is required — the single mmap buffer can only process one frame at a time
- The server must be running before you start mpv
- Seeking works but may briefly stutter as the connection recovers
- For 24fps content on a 60Hz display, you need the server to sustain ~24fps — check its output for speed

## License

RTMoSR architecture code is MIT licensed. The nvidia-vfx SDK is proprietary Nvidia software. DeH264 model weights are from their respective authors.
