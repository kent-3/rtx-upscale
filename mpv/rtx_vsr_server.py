#!/usr/bin/env python3
"""
RTX VSR Server — file-backed mmap version (works across flatpak).
Frame data via mmap files in /tmp, signaling via Unix socket.

Usage:
    python rtx_vsr_server.py [--scale 2] [--quality ULTRA] [--socket /tmp/rtx_vsr.sock]
"""

import argparse
import mmap
import os
import signal
import socket
import struct
import sys
import time

import numpy as np
import torch
import nvvfx

SHM_IN_PATH = "/tmp/rtx_vsr_in"
SHM_OUT_PATH = "/tmp/rtx_vsr_out"


def create_upscaler(out_w, out_h, quality="ULTRA"):
    ql = nvvfx.effects.QualityLevel
    quality_map = {
        name: getattr(ql, name)
        for name in dir(ql)
        if not name.startswith("_") and isinstance(getattr(ql, name), int)
    }
    q = quality_map.get(quality.upper(), ql.ULTRA)
    sr = nvvfx.VideoSuperRes(q)
    sr.output_width = out_w
    sr.output_height = out_h
    sr.load()
    return sr


@torch.inference_mode()
def process_frame(sr, mm_in, mm_out, in_h, in_w, out_h, out_w):
    """Read from mmap input, upscale, write to mmap output."""
    in_size = in_w * in_h * 3
    mm_in.seek(0)
    raw = mm_in.read(in_size)
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(3, in_h, in_w).copy()
    tensor = (
        torch.from_numpy(frame).cuda(non_blocking=True).float().div_(255.0).contiguous()
    )

    dlpack_out = sr.run(tensor).image
    output = torch.from_dlpack(dlpack_out)
    output = output.clamp_(0, 1).mul_(255.0).byte().contiguous().cpu().numpy()

    mm_out.seek(0)
    mm_out.write(output.tobytes())


def create_mmap_file(path, size):
    """Create a file and mmap it."""
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, size)
    os.close(fd)
    return mm


CONFIG_PATH = os.path.expanduser("~/.config/rtx-vsr.conf")

DEFAULT_CONFIG = {
    "quality": "ULTRA",
}


def load_config():
    """Load config from file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip()
                    if key == "quality":
                        config["quality"] = val
    return config


def save_default_config():
    """Write default config if it doesn't exist."""
    if not os.path.isfile(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            f.write("# RTX VSR Server config\n")
            f.write("# Edit and restart: systemctl --user restart rtx-vsr\n\n")
            f.write("# Quality: ULTRA, HIGH, MEDIUM, LOW\n")
            f.write(
                "#          HIGHBITRATE_ULTRA/HIGH/MEDIUM/LOW (for clean sources)\n"
            )
            f.write("#          DENOISE_ULTRA/HIGH/MEDIUM/LOW (same-res denoising)\n")
            f.write("#          DEBLUR_ULTRA/HIGH/MEDIUM/LOW (same-res deblurring)\n")
            f.write("quality=ULTRA\n")
        print(f"  Config written to {CONFIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="RTX VSR Server (mmap)")
    parser.add_argument("--quality", type=str, default=None)
    parser.add_argument("--socket", type=str, default="/tmp/rtx_vsr.sock")
    args = parser.parse_args()

    save_default_config()
    config = load_config()

    # CLI args override config file
    if args.quality is not None:
        config["quality"] = args.quality

    args.quality = config["quality"]

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    mm_in = None
    mm_out = None
    sr = None
    out_w = out_h = 0

    def cleanup(*_):
        server.close()
        for p in [sock_path, SHM_IN_PATH, SHM_OUT_PATH]:
            if os.path.exists(p):
                os.unlink(p)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"RTX VSR Server listening on {sock_path}")
    print(f"  Quality: {args.quality}")
    print(f"  Waiting for connection...")

    try:
        while True:
            conn, _ = server.accept()
            print("  Client connected.")
            frame_count = 0
            t_start = time.time()

            try:
                while True:
                    header = conn.recv(12)
                    if not header or len(header) < 12:
                        break

                    w, h, cmd = struct.unpack("<III", header)

                    if cmd == 2:  # shutdown
                        break

                    if (
                        cmd == 0
                    ):  # init — client sends 20 bytes: in_w, in_h, out_w, out_h, 0
                        extra = conn.recv(8)
                        if not extra or len(extra) < 8:
                            break
                        new_out_w, new_out_h = struct.unpack("<II", extra)

                        in_size = w * h * 3
                        out_size = new_out_w * new_out_h * 3

                        if mm_in:
                            mm_in.close()
                        if mm_out:
                            mm_out.close()

                        mm_in = create_mmap_file(SHM_IN_PATH, in_size)
                        mm_out = create_mmap_file(SHM_OUT_PATH, out_size)

                        if sr is None or new_out_w != out_w or new_out_h != out_h:
                            out_w, out_h = new_out_w, new_out_h
                            print(f"  RTX VSR: {w}x{h} -> {out_w}x{out_h}")
                            sr = create_upscaler(out_w, out_h, args.quality)

                        conn.sendall(struct.pack("<II", out_w, out_h))
                        print(f"  mmap ready: in={in_size} out={out_size}")

                    elif cmd == 1:  # process frame
                        process_frame(sr, mm_in, mm_out, h, w, out_h, out_w)
                        conn.sendall(b"\x01")

                        frame_count += 1
                        if frame_count % 100 == 0:
                            elapsed = time.time() - t_start
                            fps = frame_count / elapsed
                            print(
                                f"\r  {frame_count} frames ({fps:.1f} fps)",
                                end="",
                                flush=True,
                            )

            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"\n  Disconnected. {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)"
                )

    finally:
        cleanup()


if __name__ == "__main__":
    main()
