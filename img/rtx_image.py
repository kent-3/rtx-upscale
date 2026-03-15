#!/usr/bin/env python3
"""
rtx_image — Image enhancement using Nvidia RTX Video Super Resolution.

Supports upscaling (2x/3x/4x), denoising, and deblurring on single images
or batches. Multiple passes can be chained (e.g. denoise then upscale).

Usage:
  python rtx_image.py input.png output.png                     # 2x upscale
  python rtx_image.py input.jpg output.png --scale 4            # 4x upscale
  python rtx_image.py input.png output.png --denoise            # denoise + 2x upscale
  python rtx_image.py input.png output.png --denoise --deblur   # denoise + deblur + 2x upscale
  python rtx_image.py input.png output.png --denoise --no-upscale  # denoise only
  python rtx_image.py input_dir/ output_dir/ --scale 2          # batch process

Requires: nvidia-vfx, torch, Pillow
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ─── RTX Video Super Resolution ───────────────────────────────────────────


def create_rtx_effect(out_w: int, out_h: int, quality: str):
    """Create and configure an nvvfx VideoSuperRes context."""
    import nvvfx

    ql = nvvfx.effects.QualityLevel
    quality_map = {
        name: getattr(ql, name)
        for name in dir(ql)
        if not name.startswith("_") and isinstance(getattr(ql, name), int)
    }
    q = quality_map.get(quality.upper())
    if q is None:
        print(f"Error: Unknown quality level '{quality}'", file=sys.stderr)
        print(f"  Available: {', '.join(sorted(quality_map.keys()))}", file=sys.stderr)
        sys.exit(1)

    sr = nvvfx.VideoSuperRes(q)
    sr.output_width = out_w
    sr.output_height = out_h
    sr.load()
    return sr


@torch.inference_mode()
def run_rtx_pass(sr, tensor: torch.Tensor) -> torch.Tensor:
    """
    Run a single RTX VSR pass on a CUDA tensor.
    tensor: [3, H, W] float32 [0, 1] on CUDA, contiguous
    Returns: [3, H_out, W_out] float32 [0, 1] on CUDA
    """
    dlpack_out = sr.run(tensor).image
    return torch.from_dlpack(dlpack_out).clone()


# ─── Image I/O ─────────────────────────────────────────────────────────────


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def load_image(path: str) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB numpy array."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(path: str, data: np.ndarray):
    """Save HxWx3 uint8 RGB numpy array as an image."""
    from PIL import Image

    img = Image.fromarray(data)
    img.save(path)


@torch.inference_mode()
def image_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB numpy array to [3, H, W] float32 CUDA tensor."""
    tensor = torch.from_numpy(frame).cuda(non_blocking=True).float().div_(255.0)
    return tensor.permute(2, 0, 1).contiguous()


@torch.inference_mode()
def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert [3, H, W] float32 CUDA tensor to HxWx3 uint8 RGB numpy array."""
    return tensor.permute(1, 2, 0).clamp_(0, 1).mul_(255.0).byte().cpu().numpy()


# ─── Processing pipeline ──────────────────────────────────────────────────


@torch.inference_mode()
def process_image(
    frame: np.ndarray,
    passes: list[tuple[str, str]],
    scale: float,
    upscale_quality: str,
    no_upscale: bool,
) -> np.ndarray:
    """
    Process a single image through the RTX enhancement pipeline.
    passes: list of (pass_name, quality_level) for pre-processing passes
    """
    h, w = frame.shape[:2]
    tensor = image_to_tensor(frame)

    # Pre-processing passes (denoise, deblur) — same resolution
    for pass_name, quality in passes:
        sr = create_rtx_effect(w, h, quality)
        tensor = run_rtx_pass(sr, tensor)
        del sr

    # Upscale pass
    if not no_upscale:
        out_w = max(8, (int(w * scale) // 8) * 8)
        out_h = max(8, (int(h * scale) // 8) * 8)
        sr = create_rtx_effect(out_w, out_h, upscale_quality)
        tensor = run_rtx_pass(sr, tensor)
        del sr

    return tensor_to_image(tensor)


def build_passes(args) -> list[tuple[str, str]]:
    """Build the list of pre-processing passes from CLI args."""
    passes = []
    if args.denoise:
        passes.append(("denoise", f"DENOISE_{args.denoise_strength}"))
    if args.deblur:
        passes.append(("deblur", f"DEBLUR_{args.deblur_strength}"))
    return passes


def collect_images(path: str) -> list[Path]:
    """Collect image files from a path (single file or directory)."""
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            print(f"Error: No image files found in {path}", file=sys.stderr)
            sys.exit(1)
        return files
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RTX Image Enhancement: upscale, denoise, and deblur images using RTX Video Super Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        """,
    )

    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output image or directory")

    # Upscale options
    up = parser.add_argument_group("Upscaling")
    up.add_argument(
        "--scale", type=float, default=2.0, help="Upscale factor (default: 2.0)"
    )
    up.add_argument(
        "--quality",
        type=str,
        default="ULTRA",
        help="RTX upscale quality (default: ULTRA). "
        "Options: LOW/MEDIUM/HIGH/ULTRA, HIGHBITRATE_* (clean sources)",
    )
    up.add_argument(
        "--no-upscale",
        action="store_true",
        help="Skip upscaling (only run denoise/deblur passes)",
    )

    # Pre-processing passes
    pre = parser.add_argument_group("Pre-processing (applied before upscaling)")
    pre.add_argument(
        "--denoise",
        action="store_true",
        help="Run a denoise pass before upscaling",
    )
    pre.add_argument(
        "--denoise-strength",
        type=str,
        default="ULTRA",
        choices=["LOW", "MEDIUM", "HIGH", "ULTRA"],
        help="Denoise strength (default: ULTRA)",
    )
    pre.add_argument(
        "--deblur",
        action="store_true",
        help="Run a deblur pass before upscaling",
    )
    pre.add_argument(
        "--deblur-strength",
        type=str,
        default="ULTRA",
        choices=["LOW", "MEDIUM", "HIGH", "ULTRA"],
        help="Deblur strength (default: ULTRA)",
    )

    # Output options
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--format",
        type=str,
        default=None,
        help="Output format (png, jpg, webp). Auto-detected from extension if omitted",
    )

    args = parser.parse_args()

    # Validate
    if args.no_upscale and not args.denoise and not args.deblur:
        print(
            "Error: --no-upscale requires --denoise and/or --deblur",
            file=sys.stderr,
        )
        sys.exit(1)

    passes = build_passes(args)

    # Collect input images
    input_files = collect_images(args.input)
    is_batch = len(input_files) > 1 or Path(args.input).is_dir()

    # Resolve output paths
    if is_batch:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for f in input_files:
            ext = f".{args.format}" if args.format else f.suffix
            output_files.append(out_dir / f"{f.stem}{ext}")
    else:
        output_files = [Path(args.output)]
        output_files[0].parent.mkdir(parents=True, exist_ok=True)

    # Describe pipeline
    pipeline_desc = []
    for name, quality in passes:
        pipeline_desc.append(f"{name} ({quality})")
    if not args.no_upscale:
        pipeline_desc.append(f"{args.scale}x upscale ({args.quality})")
    print(f"Pipeline: {' → '.join(pipeline_desc)}")
    print(
        f"Processing {len(input_files)} image{'s' if len(input_files) != 1 else ''}..."
    )

    # Process
    t_start = time.time()
    for i, (in_path, out_path) in enumerate(zip(input_files, output_files)):
        t_img = time.time()
        frame = load_image(str(in_path))
        h, w = frame.shape[:2]

        result = process_image(frame, passes, args.scale, args.quality, args.no_upscale)
        save_image(str(out_path), result)

        oh, ow = result.shape[:2]
        elapsed = time.time() - t_img
        print(
            f"  [{i + 1}/{len(input_files)}] {in_path.name}: "
            f"{w}x{h} → {ow}x{oh} ({elapsed:.2f}s)"
        )

    total = time.time() - t_start
    print(
        f"\nDone! {len(input_files)} image{'s' if len(input_files) != 1 else ''} "
        f"in {total:.2f}s"
    )


if __name__ == "__main__":
    main()
