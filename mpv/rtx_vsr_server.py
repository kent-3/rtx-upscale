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


def create_upscaler(out_w, out_h, quality="HIGHBITRATE_ULTRA"):
    ql = nvvfx.effects.QualityLevel
    quality_map = {
        name: getattr(ql, name)
        for name in dir(ql)
        if not name.startswith("_") and isinstance(getattr(ql, name), int)
    }
    q = quality_map.get(quality.upper(), ql.HIGHBITRATE_ULTRA)
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


def main():
    parser = argparse.ArgumentParser(description="RTX VSR Server (mmap)")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--quality",
        type=str,
        default="HIGHBITRATE_ULTRA",
    )
    parser.add_argument("--socket", type=str, default="/tmp/rtx_vsr.sock")
    args = parser.parse_args()

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
    print(f"  Scale: {args.scale}x | Quality: {args.quality}")
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

                    if cmd == 0:  # init
                        new_out_w = max(8, int(w * args.scale) // 8 * 8)
                        new_out_h = max(8, int(h * args.scale) // 8 * 8)

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
