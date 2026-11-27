#!/usr/bin/env python3
"""AI Photo HD Enhancer worker using Real-ESRGAN + GFPGAN v1.4 on CPU."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import tempfile
import urllib.request

import boto3
from botocore.config import Config
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

WEIGHTS_DIR = Path(os.environ.get("AI_WEIGHTS_DIR", str(Path.home() / ".cache" / "ai_weights")))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "general_fast": {
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "netscale": 4,
    },
    "general_hd": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "netscale": 2,
    },
    "anime": {
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "netscale": 4,
    },
}

GFPGAN_MODEL = {
    "filename": "GFPGANv1.4.pth",
    "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
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


def ensure_weight(filename: str, url: str) -> Path:
    dest = WEIGHTS_DIR / filename
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def build_upsampler(style: str) -> RealESRGANer:
    spec = MODEL_REGISTRY.get(style, MODEL_REGISTRY["general_fast"])
    w_path = ensure_weight(spec["filename"], spec["url"])

    if style == "general_hd":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif style == "anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    else:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")

    return RealESRGANer(
        scale=spec["netscale"],
        model_path=str(w_path),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch.device("cpu"),
    )


def enhance_photo(src_path: Path, out_path: Path, style: str, outscale: int, face_enhance: bool, out_fmt: str) -> str:
    # Normalize EXIF orientation before OpenCV processing
    with Image.open(src_path) as opened:
        icc = opened.info.get("icc_profile")
        pil_img = ImageOps.exif_transpose(opened).convert("RGB")

    # Guard against massive camera originals so CPU runner finishes reliably within minutes
    max_input = 1600 if outscale <= 2 else 1024
    if max(pil_img.size) > max_input:
        ratio = max_input / max(pil_img.size)
        pil_img = pil_img.resize(
            (max(1, round(pil_img.width * ratio)), max(1, round(pil_img.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    upsampler = build_upsampler(style)

    if face_enhance and style != "anime":
        gfp_path = ensure_weight(GFPGAN_MODEL["filename"], GFPGAN_MODEL["url"])
        restorer = GFPGANer(
            model_path=str(gfp_path),
            upscale=outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=torch.device("cpu"),
        )
        _, _, output_bgr = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.6,
        )
    else:
        output_bgr, _ = upsampler.enhance(bgr, outscale=outscale)

    rgb_out = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(rgb_out)

    save_kwargs = {"icc_profile": icc} if icc else {}
    if out_fmt == "png":
        out_img.save(out_path, format="PNG", optimize=True, compress_level=6, **save_kwargs)
    else:
        out_img.save(out_path, format="JPEG", quality=92, optimize=True, progressive=True, subsampling=0, **save_kwargs)

    face_note = " + perbaiki wajah" if (face_enhance and style != "anime") else ""
    return f"HD {outscale}x ({out_img.width}x{out_img.height}){face_note}"


def run_job(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    status_key = f"jobs/{job_id}/status.json"

    raw = s3.get_object(Bucket=bucket, Key=status_key)["Body"].read()
    job = json.loads(raw.decode("utf-8"))
    job.update({"status": "processing", "progress": 10, "speed": ""})
    write_status(s3, bucket, job_id, job)

    try:
        opts = job.get("options") or {}
        style = str(opts.get("style", "general_fast"))
        outscale = 4 if int(opts.get("scale", 2)) >= 4 else 2
        face_enhance = bool(opts.get("faceEnhance", True))
        out_fmt = "png" if str(opts.get("format", "jpg")).lower() == "png" else "jpg"
        ext = ".png" if out_fmt == "png" else ".jpg"

        items = job.get("items") or [{"inputKey": job["inputKey"], "filename": job.get("filename") or Path(job["inputKey"]).name}]
        item = items[0]
        in_key = item["inputKey"]
        orig_name = Path(item.get("filename") or in_key).name

        with tempfile.TemporaryDirectory(prefix=f"enh_{job_id}_") as td:
            work = Path(td)
            src_path = work / orig_name
            s3.download_file(bucket, in_key, str(src_path))
            orig_bytes = src_path.stat().st_size

            job.update({"progress": 25})
            write_status(s3, bucket, job_id, job)

            out_name = f"{src_path.stem}_HD{outscale}x{ext}"
            out_path = work / out_name
            detail = enhance_photo(src_path, out_path, style, outscale, face_enhance, out_fmt)

            job.update({"progress": 90})
            write_status(s3, bucket, job_id, job)

            output_key = f"jobs/{job_id}/output/{out_name}"
            s3.upload_file(str(out_path), bucket, output_key)
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
                    "detail": detail,
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
