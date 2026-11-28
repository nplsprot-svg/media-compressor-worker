#!/usr/bin/env python3
"""Ultra AI Photo Enhancer worker using InvSR (Diffusion Restoration on ZeroGPU) with Finegrain fallback."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import tempfile

import boto3
from botocore.config import Config
from gradio_client import Client, handle_file
from PIL import Image, ImageOps

INVSR_SPACE = "OAOA/InvSR"
FINEGRAIN_SPACE = "finegrain/finegrain-image-enhancer"


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


def extract_output_path(result) -> Path:
    candidates = []
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, (list, tuple)):
                candidates.extend(item)
            else:
                candidates.append(item)
    else:
        candidates.append(result)

    for item in reversed(candidates):
        path_str = item.get("path") or item.get("url") if isinstance(item, dict) else item
        if isinstance(path_str, str):
            p = Path(path_str)
            if p.exists() and p.is_file():
                return p
    raise RuntimeError(f"Format output tidak dikenali: {type(result)}")


def run_invsr(prep_path: Path, hf_token: str | None) -> Path:
    client = Client(INVSR_SPACE, token=hf_token)
    res = client.predict(
        in_path=handle_file(str(prep_path)),
        num_steps=3,
        chopping_size=128,
        seed=12345,
        api_name="/predict",
    )
    return extract_output_path(res)


def run_finegrain(prep_path: Path, gpu_upscale: int, hf_token: str | None) -> Path:
    client = Client(FINEGRAIN_SPACE, token=hf_token)
    res = client.predict(
        input_image=handle_file(str(prep_path)),
        prompt="sharp focus, clean anti-aliased edges, faithful high resolution restoration",
        negative_prompt="blurry, jagged edges, aliasing, pixelated, artifacts, distorted text, lowres",
        seed=42,
        upscale_factor=gpu_upscale,
        controlnet_scale=0.84,
        controlnet_decay=1.0,
        condition_scale=4.2,
        tile_width=112,
        tile_height=144,
        denoise_strength=0.23,
        num_inference_steps=20,
        solver="DDIM",
        api_name="/process",
    )
    return extract_output_path(res)


def run_job(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    s3 = make_s3_client()
    status_key = f"jobs/{job_id}/status.json"

    raw = s3.get_object(Bucket=bucket, Key=status_key)["Body"].read()
    job = json.loads(raw.decode("utf-8"))
    job.update({"status": "processing", "progress": 15, "speed": ""})
    write_status(s3, bucket, job_id, job)

    try:
        opts = job.get("options") or {}
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

        with tempfile.TemporaryDirectory(prefix=f"ultra_{job_id}_") as td:
            work = Path(td)
            src_path = work / orig_name
            s3.download_file(bucket, in_key, str(src_path))
            orig_bytes = src_path.stat().st_size

            with Image.open(src_path) as opened:
                icc = opened.info.get("icc_profile")
                pil_img = ImageOps.exif_transpose(opened).convert("RGB")

            orig_w, orig_h = pil_img.size
            target_w = orig_w * req_scale
            target_h = orig_h * req_scale

            # InvSR natively upscales 4x; pre-scaling to ~0.78x (capped at 768px) melts 2-3px game jaggies
            # before 4x diffusion restoration, then Lanczos supersamples cleanly to target 1x/2x/4x.
            prep_img = pil_img
            max_prep = 768
            if max(prep_img.size) > max_prep:
                ratio = max_prep / max(prep_img.size)
                prep_img = prep_img.resize(
                    (max(64, round(prep_img.width * ratio)), max(64, round(prep_img.height * ratio))),
                    Image.Resampling.HAMMING,
                )
            else:
                prep_img = prep_img.resize(
                    (max(64, round(prep_img.width * 0.78)), max(64, round(prep_img.height * 0.78))),
                    Image.Resampling.HAMMING,
                )

            prep_path = work / "input_prep.png"
            prep_img.save(prep_path, format="PNG")

            job.update({"progress": 35})
            write_status(s3, bucket, job_id, job)

            engine_label = "Ultra AI"
            try:
                raw_out = run_invsr(prep_path, hf_token)
            except Exception as inv_err:
                if "quota" in str(inv_err).lower() or "exceeded" in str(inv_err).lower():
                    raise
                gpu_upscale = 4 if req_scale == 4 else 2
                raw_out = run_finegrain(prep_path, gpu_upscale, hf_token)
                engine_label = "Ultra AI (Alt)"

            job.update({"progress": 85})
            write_status(s3, bucket, job_id, job)

            with Image.open(raw_out) as enhanced:
                out_img = enhanced.convert("RGB")
                if out_img.size != (target_w, target_h):
                    out_img = out_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            suffix = "HD" if req_scale == 1 else f"HD{req_scale}x"
            out_name = f"{src_path.stem}_{suffix}{ext}"
            out_path = work / out_name
            save_kwargs = {"icc_profile": icc} if icc else {}
            if out_fmt == "png":
                out_img.save(out_path, format="PNG", optimize=True, compress_level=6, **save_kwargs)
            else:
                out_img.save(out_path, format="JPEG", quality=93, optimize=True, progressive=True, subsampling=0, **save_kwargs)

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
                    "detail": f"{engine_label} {req_scale}x ({out_img.width}x{out_img.height})",
                    "isAudioOnly": False,
                    "isAnimatedImage": False,
                }
            )
            write_status(s3, bucket, job_id, job)
    except Exception as exc:
        msg = str(exc)
        if "quota" in msg.lower() or "exceeded" in msg.lower():
            msg = "Kuota GPU AI harian sedang penuh. Gunakan mode Foto Halus & Tajam atau Game 3D & Anime yang gratis tanpa batas."
        job.update({"status": "error", "error": msg})
        write_status(s3, bucket, job_id, job)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run_job(args.job_id)


if __name__ == "__main__":
    main()
