#!/usr/bin/env python3
"""AI RIFE (ONNX) 60fps Video Frame Interpolator supporting RIFE v4.17 Lite (fast) and RIFE v4.9 Ensemble (max quality)."""

from pathlib import Path
import math
import os
import subprocess
import urllib.request

import numpy as np
import onnxruntime as ort

WEIGHTS_DIR = Path(os.environ.get("AI_WEIGHTS_DIR", str(Path.home() / ".cache" / "ai_weights_rife")))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

RIFE_MODELS = {
    "fast": {
        "filename": "rife-v4.17-lite.onnx",
        "url": "https://huggingface.co/archgod94/tellcue-models/resolve/main/rife-v4.17-lite.onnx",
        "label": "AI RIFE 60fps",
    },
    "max": {
        "filename": "rife49_ensemble_True_scale_1_sim.onnx",
        "url": "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx",
        "label": "AI RIFE Ensemble 60fps",
    },
}
SCENE_CUT_THRESHOLD = 0.22


def ensure_rife_model(quality: str = "fast") -> tuple[Path, str]:
    spec = RIFE_MODELS.get(quality, RIFE_MODELS["fast"])
    dest = WEIGHTS_DIR / spec["filename"]
    if dest.exists() and dest.stat().st_size > 5_000_000:
        return dest, spec["label"]
    tmp = dest.with_suffix(".tmp")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=240) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    return dest, spec["label"]


def _pad_to_mult(arr_chw: np.ndarray, mult: int) -> tuple[np.ndarray, int, int]:
    _, h, w = arr_chw.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return arr_chw, 0, 0
    padded = np.pad(arr_chw, ((0, 0), (0, ph), (0, pw)), mode="edge")
    return padded, ph, pw


def _is_scene_cut(f0_u8: np.ndarray, f1_u8: np.ndarray) -> bool:
    s0 = f0_u8[::8, ::8, :].astype(np.float32)
    s1 = f1_u8[::8, ::8, :].astype(np.float32)
    mad = float(np.mean(np.abs(s1 - s0))) / 255.0
    return mad > SCENE_CUT_THRESHOLD


def interpolate_video_60fps(
    src_path: Path,
    dst_path: Path,
    width: int,
    height: int,
    src_fps: float,
    duration: float,
    pre_vf: list[str],
    trim_args: list[str],
    codec_args: list[str],
    audio_args: list[str],
    extra_mux_args: list[str],
    ffmpeg_bin: str = "ffmpeg",
    progress_cb=None,
    rife_quality: str = "fast",
    drop_last_frame: bool = False,
) -> str:
    """Decode pre-filtered frames -> AI RIFE 60fps -> pipe straight into FFmpeg encoder with audio."""
    model_path, label = ensure_rife_model(rife_quality)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    in_img0 = inputs[0].name
    in_img1 = inputs[1].name
    in_ts = inputs[2].name
    has_scale = len(inputs) >= 4
    in_scale = inputs[3].name if has_scale else None
    out_name = sess.get_outputs()[0].name

    dst_fps = 60.0
    eff_fps = src_fps if src_fps > 1.0 else 30.0
    flow_scale = 0.5 if (has_scale and max(width, height) >= 720) else 1.0
    pad_mult = int(round(32.0 / flow_scale)) if has_scale else 32
    scale_arr = np.array([flow_scale], dtype=np.float32)

    dec_vf = list(pre_vf)
    dec_cmd = [ffmpeg_bin, "-hide_banner", "-v", "error", *trim_args, "-i", str(src_path)]
    if dec_vf:
        dec_cmd += ["-vf", ",".join(dec_vf)]
    dec_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    enc_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{dst_fps:.3f}",
        "-i",
        "pipe:0",
        *trim_args,
        "-i",
        str(src_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        *codec_args,
        *audio_args,
        *extra_mux_args,
        str(dst_path),
    ]

    frame_bytes = width * height * 3
    est_in_frames = max(2, int(round(max(duration, 0.5) * eff_fps)))
    est_out_frames = max(2, int(round((est_in_frames - 1) * dst_fps / eff_fps)) + 1)

    dec_proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def read_frame() -> np.ndarray | None:
        buf = dec_proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))

    def to_tensor(f_u8: np.ndarray) -> np.ndarray:
        chw = np.transpose(f_u8.astype(np.float32) / 255.0, (2, 0, 1))
        padded, _, _ = _pad_to_mult(chw, pad_mult)
        return padded[None, ...]

    try:
        prev_u8 = read_frame()
        if prev_u8 is None:
            raise RuntimeError("Gagal membaca frame video untuk interpolasi 60fps.")
        prev_t = None
        curr_u8 = read_frame()
        curr_t = None

        if curr_u8 is None:
            enc_proc.stdin.write(prev_u8.tobytes())
        else:
            in_idx = 0
            out_idx = 0
            while True:
                pos = out_idx * eff_fps / dst_fps
                target_idx = int(math.floor(pos))
                alpha = pos - target_idx

                while in_idx < target_idx:
                    nxt = read_frame()
                    if nxt is None:
                        break
                    prev_u8, prev_t = curr_u8, curr_t
                    curr_u8, curr_t = nxt, None
                    in_idx += 1

                if target_idx > in_idx:
                    if not drop_last_frame:
                        enc_proc.stdin.write(curr_u8.tobytes())
                    break

                if alpha < 0.02:
                    enc_proc.stdin.write(prev_u8.tobytes())
                elif alpha > 0.98:
                    enc_proc.stdin.write(curr_u8.tobytes())
                elif _is_scene_cut(prev_u8, curr_u8):
                    enc_proc.stdin.write(prev_u8.tobytes())
                else:
                    if prev_t is None:
                        prev_t = to_tensor(prev_u8)
                    if curr_t is None:
                        curr_t = to_tensor(curr_u8)
                    ts_arr = np.array([alpha], dtype=np.float32)
                    feed = {in_img0: prev_t, in_img1: curr_t, in_ts: ts_arr}
                    if has_scale and in_scale:
                        feed[in_scale] = scale_arr
                    pred = sess.run([out_name], feed)[0][0]
                    pred_hwc = np.transpose(pred[:, :height, :width], (1, 2, 0))
                    out_u8 = np.clip(pred_hwc * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    enc_proc.stdin.write(out_u8.tobytes())

                out_idx += 1
                if progress_cb and (out_idx % 2 == 0):
                    pct = min(98.0, (out_idx / est_out_frames) * 100.0)
                    progress_cb(pct, label)
    finally:
        if enc_proc.stdin:
            try:
                enc_proc.stdin.close()
            except Exception:
                pass
        if dec_proc.stdout:
            try:
                dec_proc.stdout.close()
            except Exception:
                pass

    dec_err = dec_proc.stderr.read().decode("utf-8", errors="replace").strip() if dec_proc.stderr else ""
    enc_err = enc_proc.stderr.read().decode("utf-8", errors="replace").strip() if enc_proc.stderr else ""
    dec_rc = dec_proc.wait()
    enc_rc = enc_proc.wait()
    if dec_rc != 0:
        raise RuntimeError(f"Decode video gagal: {dec_err[-240:]}")
    if enc_rc != 0:
        raise RuntimeError(f"Encode 60fps gagal: {enc_err[-240:]}")
    return label
