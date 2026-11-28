#!/usr/bin/env python3
"""Unlimited CPU ONNX Photo HD Enhancer: Selective Structure-Tensor FXAA + 4x-UltraSharp / 4x-UltraSharpV2 Full (DAT2) + Subtle LAB Clarity."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import tempfile
import urllib.request

import boto3
from botocore.config import Config
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

WEIGHTS_DIR = Path(os.environ.get("AI_WEIGHTS_DIR", str(Path.home() / ".cache" / "ai_weights_onnx")))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

ONNX_MODELS = {
    "ultrasharp_aa": {
        "filename": "4x-UltraSharp-fp32-opset17.onnx",
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/ONNX/4x-UltraSharp-fp32-opset17.onnx",
        "label": "UltraSharp Pro",
    },
    "ultrasharp_v2_max": {
        "filename": "4x-UltraSharpV2_fp32_op17.onnx",
        "url": "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2_fp32_op17.onnx",
        "label": "UltraSharp V2 Max",
    },
}


def make_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "").strip()
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-005"),
        config=Config(signature_version="s3v4"),
    )


def write_status(s3, bucket: str, job_id: str, payload: dict) -> None:
    payload["updatedAt"] = datetime.now(timezone.utc).isoformat()
    s3.put_object(
        Bucket=bucket,
        Key=f"jobs/{job_id}/status.json",
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def ensure_model(style: str) -> tuple[Path, str]:
    spec = ONNX_MODELS.get(style, ONNX_MODELS["ultrasharp_aa"])
    dest = WEIGHTS_DIR / spec["filename"]
    if dest.exists() and dest.stat().st_size > 10_000_000:
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


def fxaa_pass(rgb_u8: np.ndarray, span_max: float = 9.0, reduce_mul: float = 1.0 / 8.0, reduce_min: float = 1.0 / 128.0) -> np.ndarray:
    """NVIDIA FXAA 3.11 edge-directed anti-aliasing in vectorized NumPy/OpenCV."""
    img = rgb_u8.astype(np.float32) / 255.0
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

    def luma(x: np.ndarray) -> np.ndarray:
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    luma_nw = luma(p[:-2, :-2])
    luma_ne = luma(p[:-2, 2:])
    luma_sw = luma(p[2:, :-2])
    luma_se = luma(p[2:, 2:])
    luma_m = luma(img)

    luma_min = np.minimum(luma_m, np.minimum(np.minimum(luma_nw, luma_ne), np.minimum(luma_sw, luma_se)))
    luma_max = np.maximum(luma_m, np.maximum(np.maximum(luma_nw, luma_ne), np.maximum(luma_sw, luma_se)))
    luma_range = luma_max - luma_min

    dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se))
    dir_y = (luma_nw + luma_sw) - (luma_ne + luma_se)

    dir_reduce = np.maximum((luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * reduce_mul), reduce_min)
    rcp_dir_min = 1.0 / (np.minimum(np.abs(dir_x), np.abs(dir_y)) + dir_reduce)

    dx = np.clip(dir_x * rcp_dir_min, -span_max, span_max)
    dy = np.clip(dir_y * rcp_dir_min, -span_max, span_max)

    h, w = luma_m.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    def sample(off: float) -> np.ndarray:
        return cv2.remap(
            img,
            (xx + dx * off).astype(np.float32),
            (yy + dy * off).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    rgb_a = 0.5 * (sample(1.0 / 3.0 - 0.5) + sample(2.0 / 3.0 - 0.5))
    rgb_b = rgb_a * 0.5 + 0.25 * (sample(-0.5) + sample(0.5))

    luma_b = luma(rgb_b)
    use_a = (luma_b < luma_min) | (luma_b > luma_max)
    fxaa = np.where(use_a[..., None], rgb_a, rgb_b)

    active = luma_range >= np.maximum(0.05, luma_max * 0.125)
    out = np.where(active[..., None], fxaa, img)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def selective_edge_aa(rgb_u8: np.ndarray) -> np.ndarray:
    """Apply 2-pass FXAA selectively on high-coherence diagonal edges, keeping fine text & textures razor-sharp."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (7, 7), 1.6)
    jyy = cv2.GaussianBlur(gy * gy, (7, 7), 1.6)
    jxy = cv2.GaussianBlur(gx * gy, (7, 7), 1.6)

    trace = jxx + jyy + 1e-6
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    coherence = diff / trace

    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    diag_weight = np.clip(np.abs(np.sin(2.0 * angle)) * 2.8, 0.0, 1.0)
    coh_mask = np.clip((coherence - 0.52) / 0.28, 0.0, 1.0)
    mask = cv2.GaussianBlur(np.clip(coh_mask * diag_weight, 0.0, 1.0), (3, 3), 0.7)[..., None]

    aa_img = fxaa_pass(fxaa_pass(rgb_u8, span_max=9.0), span_max=9.0)
    blended = rgb_u8.astype(np.float32) * (1.0 - mask) + aa_img.astype(np.float32) * mask
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)


def subtle_clarity_polish(rgb_u8: np.ndarray) -> np.ndarray:
    """Clean luminance-only crispness & clarity polish (no harsh contrast, no color shift, no halos)."""
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    blur_fine = cv2.GaussianBlur(L, (0, 0), sigmaX=1.0)
    detail = np.clip((L - blur_fine) * 0.35, -14.0, 14.0)
    blur_wide = cv2.GaussianBlur(L, (0, 0), sigmaX=10.0)
    clarity = np.clip((L - blur_wide) * 0.08, -10.0, 10.0)
    lab[..., 0] = np.clip(L + detail + clarity, 0.0, 255.0)
    lab[..., 1] = np.clip(128.0 + (lab[..., 1] - 128.0) * 1.03, 0.0, 255.0)
    lab[..., 2] = np.clip(128.0 + (lab[..., 2] - 128.0) * 1.03, 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def upscale_onnx_tiled(rgb_u8: np.ndarray, sess: ort.InferenceSession, on_tile=None, tile: int = 256, pad: int = 16) -> Image.Image:
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    arr = rgb_u8.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    out = np.zeros((h * 4, w * 4, 3), dtype=np.float32)

    ys = list(range(0, h, tile))
    xs = list(range(0, w, tile))
    total_tiles = len(ys) * len(xs)
    done_tiles = 0

    for y in ys:
        for x in xs:
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
            inp = np.transpose(arr[y0:y1, x0:x1, :], (2, 0, 1))[None, ...]
            pred = sess.run([out_name], {in_name: inp})[0][0]
            pred = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)

            cy0, cx0 = (y - y0) * 4, (x - x0) * 4
            cy1 = cy0 + min(tile, h - y) * 4
            cx1 = cx0 + min(tile, w - x) * 4
            out[y * 4 : min(h, y + tile) * 4, x * 4 : min(w, x + tile) * 4, :] = pred[cy0:cy1, cx0:cx1, :]

            done_tiles += 1
            if on_tile:
                on_tile(done_tiles, total_tiles)

    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


def run_job(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    status_key = f"jobs/{job_id}/status.json"

    raw = s3.get_object(Bucket=bucket, Key=status_key)["Body"].read()
    job = json.loads(raw.decode("utf-8"))
    job.update({"status": "processing", "progress": 10, "speed": ""})
    write_status(s3, bucket, job_id, job)

    def handle_sigterm(_signum, _frame):
        job.update({"status": "error", "error": "Proses HD foto melebihi batas waktu server (30 menit). Coba gunakan mode UltraSharp Pro."})
        try:
            write_status(s3, bucket, job_id, job)
        except Exception:
            pass
        sys.exit(143)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        opts = job.get("options") or {}
        style = str(opts.get("style", "ultrasharp_aa")).lower()
        req_scale = int(opts.get("scale", 2))
        if req_scale not in (1, 2, 4):
            req_scale = 2
        out_fmt = "png" if str(opts.get("format", "jpg")).lower() == "png" else "jpg"
        ext = ".png" if out_fmt == "png" else ".jpg"
        mime = "image/png" if out_fmt == "png" else "image/jpeg"

        items = job.get("items") or [{"inputKey": job["inputKey"], "filename": job.get("filename") or Path(job["inputKey"]).name}]
        item = items[0]
        in_key = item["inputKey"]
        orig_name = Path(item.get("filename") or in_key).name

        model_path, label = ensure_model(style)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

        with tempfile.TemporaryDirectory(prefix=f"onnx_{job_id}_") as td:
            work = Path(td)
            src_path = work / orig_name
            s3.download_file(bucket, in_key, str(src_path))
            orig_bytes = src_path.stat().st_size

            with Image.open(src_path) as opened:
                icc = opened.info.get("icc_profile")
                pil_img = ImageOps.exif_transpose(opened).convert("RGB")

            max_input = 1440 if req_scale <= 2 else 1080
            if max(pil_img.size) > max_input:
                ratio = max_input / max(pil_img.size)
                pil_img = pil_img.resize(
                    (max(64, round(pil_img.width * ratio)), max(64, round(pil_img.height * ratio))),
                    Image.Resampling.LANCZOS,
                )

            target_w = pil_img.width * req_scale
            target_h = pil_img.height * req_scale

            job.update({"progress": 20})
            write_status(s3, bucket, job_id, job)

            pre_rgb = selective_edge_aa(np.asarray(pil_img))

            last_pct = 20

            def on_tile(done_t: int, total_t: int) -> None:
                nonlocal last_pct
                pct = 22 + int((done_t / max(1, total_t)) * 68)
                if pct >= last_pct + 6 or done_t == total_t:
                    last_pct = pct
                    job.update({"progress": pct})
                    write_status(s3, bucket, job_id, job)

            tile_size = 224 if style == "ultrasharp_v2_max" else 256
            out_img = upscale_onnx_tiled(pre_rgb, sess, on_tile=on_tile, tile=tile_size)
            if out_img.size != (target_w, target_h):
                out_img = out_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            out_img = Image.fromarray(subtle_clarity_polish(np.asarray(out_img)))

            suffix = "HD" if req_scale == 1 else f"HD{req_scale}x"
            out_name = f"{src_path.stem}_{suffix}{ext}"
            out_path = work / out_name
            save_kwargs = {"icc_profile": icc} if icc else {}
            if out_fmt == "png":
                out_img.save(out_path, format="PNG", optimize=True, compress_level=6, **save_kwargs)
            else:
                out_img.save(out_path, format="JPEG", quality=94, optimize=True, progressive=True, subsampling=0, **save_kwargs)

            output_key = f"jobs/{job_id}/output/{out_name}"
            s3.upload_file(str(out_path), bucket, output_key, ExtraArgs={"ContentType": mime})
            new_bytes = out_path.stat().st_size

            job.update(
                {
                    "status": "done",
                    "progress": 100,
                    "isBatch": False,
                    "outputKey": output_key,
                    "outputName": out_name,
                    "origBytes": orig_bytes,
                    "newBytes": new_bytes,
                    "detail": f"{label} {req_scale}x ({out_img.width}x{out_img.height})",
                    "isAudioOnly": False,
                    "isAnimatedImage": False,
                }
            )
            write_status(s3, bucket, job_id, job)
    except Exception as exc:
        job.update({"status": "error", "error": str(exc)})
        write_status(s3, bucket, job_id, job)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run_job(args.job_id)


if __name__ == "__main__":
    main()
