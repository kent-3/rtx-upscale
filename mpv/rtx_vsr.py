"""
VapourSynth filter for mpv — RTX VSR via file-backed mmap (zero-copy, flatpak-safe).

Terminal 1:
    /home/kent/ai/ComfyUI/.venv/bin/python /home/kent/ai/rtx-upscale-cli/mpv/rtx_vsr_server.py

Terminal 2:
    mpv video.mp4 --hwdec=no --vf="vapoursynth=file=/home/kent/ai/rtx-upscale-cli/mpv/rtx_vsr.py:concurrent-frames=1"
"""

import ctypes
import mmap
import os
import socket
import struct
import threading
import vapoursynth as vs

SOCKET_PATH = "/tmp/rtx_vsr.sock"
SHM_IN_PATH = "/tmp/rtx_vsr_in"
SHM_OUT_PATH = "/tmp/rtx_vsr_out"
SCALE = 2

core = vs.core

_conn = None
_lock = threading.Lock()
_mm_in = None
_mm_out = None
_out_w = 0
_out_h = 0
_initialized = False


def ensure_connection():
    global _conn
    if _conn is None:
        _conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _conn.connect(SOCKET_PATH)
    return _conn


def init_shm(width, height):
    """Send init command, wait for server to create mmap files, then open them."""
    global _mm_in, _mm_out, _out_w, _out_h, _initialized

    conn = ensure_connection()
    conn.sendall(struct.pack("<III", width, height, 0))

    resp = conn.recv(8)
    _out_w, _out_h = struct.unpack("<II", resp)

    # Open the mmap files created by the server
    in_size = width * height * 3
    out_size = _out_w * _out_h * 3

    fd_in = os.open(SHM_IN_PATH, os.O_RDWR)
    _mm_in = mmap.mmap(fd_in, in_size)
    os.close(fd_in)

    fd_out = os.open(SHM_OUT_PATH, os.O_RDONLY)
    _mm_out = mmap.mmap(fd_out, out_size, prot=mmap.PROT_READ)
    os.close(fd_out)

    _initialized = True


def _reconnect(width, height):
    """Reset connection and re-initialize mmap."""
    global _conn, _mm_in, _mm_out, _initialized
    try:
        if _conn:
            _conn.close()
    except Exception:
        pass
    _conn = None
    try:
        if _mm_in:
            _mm_in.close()
    except Exception:
        pass
    try:
        if _mm_out:
            _mm_out.close()
    except Exception:
        pass
    _mm_in = None
    _mm_out = None
    _initialized = False
    init_shm(width, height)


def make_frame(n, f):
    global _initialized

    src = f[0]
    width = src.width
    height = src.height

    with _lock:
        if not _initialized:
            init_shm(width, height)

        try:
            # Write planar RGB into mmap
            plane_size = width * height
            _mm_in.seek(0)
            _mm_in.write(bytes(src[0]))
            _mm_in.write(bytes(src[1]))
            _mm_in.write(bytes(src[2]))

            # Signal server
            conn = ensure_connection()
            conn.sendall(struct.pack("<III", width, height, 1))
            resp = conn.recv(1)  # wait for done
            if not resp:
                raise ConnectionError("Server closed connection")

        except (BrokenPipeError, ConnectionError, OSError):
            # Reconnect and retry on seek/disconnect
            _reconnect(width, height)
            _mm_in.seek(0)
            _mm_in.write(bytes(src[0]))
            _mm_in.write(bytes(src[1]))
            _mm_in.write(bytes(src[2]))
            conn = ensure_connection()
            conn.sendall(struct.pack("<III", width, height, 1))
            conn.recv(1)

        # Read output from mmap into frame
        out_plane_size = _out_w * _out_h
        fout = f[1].copy()

        for p in range(3):
            ptr = ctypes.cast(fout.get_write_ptr(p), ctypes.c_void_p).value
            stride = fout.get_stride(p)
            src_offset = p * out_plane_size

            _mm_out.seek(src_offset)
            plane_data = _mm_out.read(out_plane_size)

            if stride == _out_w:
                ctypes.memmove(ptr, plane_data, out_plane_size)
            else:
                for row in range(_out_h):
                    ctypes.memmove(
                        ptr + row * stride,
                        plane_data[row * _out_w : (row + 1) * _out_w],
                        _out_w,
                    )

    fout.props.update(src.props)
    return fout


# ─── Filter setup ──────────────────────────────────────────────────────────

clip = video_in  # noqa: F821

clip_rgb = core.resize.Bilinear(clip, format=vs.RGB24, matrix_in_s="709")

out_w = max(8, (clip.width * SCALE) // 8 * 8)
out_h = max(8, (clip.height * SCALE) // 8 * 8)

blank = core.std.BlankClip(clip_rgb, width=out_w, height=out_h, format=vs.RGB24)
clip_out = core.std.ModifyFrame(blank, [clip_rgb, blank], make_frame)
clip_out = core.resize.Bilinear(clip_out, format=vs.YUV420P8, matrix_s="709")

clip_out.set_output()
