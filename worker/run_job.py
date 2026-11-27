#!/usr/bin/env python3
"""Media compression worker: downloads input from S3/B2, streams progress, uploads compressed output."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import time

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_compress as pc
import video_compress as vc


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


def run_job(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    status_key = f"jobs/{job_id}/status.json"

    raw = s3.get_object(Bucket=bucket, Key=status_key)["Body"].read()
    job = json.loads(raw.decode("utf-8"))
    job.update({"status": "processing", "progress": 2, "speed": ""})
    write_status(s3, bucket, job_id, job)

    last_push = 0.0

    def on_progress(pct: float, speed: str) -> None:
        nonlocal last_push
        now = time.monotonic()
        if now - last_push < 3.0:
            return
        last_push = now
        job["progress"] = round(min(99.0, max(1.0, pct)), 1)
        job["speed"] = speed
        try:
            write_status(s3, bucket, job_id, job)
        except Exception:
            pass

    try:
        opts = job.get("options") or {}
        mode = job["mode"]
        input_key = job["inputKey"]
        filename = Path(job.get("filename") or input_key).name

        with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as td:
            work = Path(td)
            src_path = work / filename
            out_dir = work / "out"
            out_dir.mkdir()

            s3.download_file(bucket, input_key, str(src_path))

            if mode == "photo":
                cfg = pc.CompressConfig(
                    output_format=str(opts.get("format", "keep")),
                    quality=int(opts.get("quality", 85)),
                    target_kb=int(opts.get("targetKb", 0)),
                    max_side=int(opts.get("maxSide", 0)),
                    resample=str(opts.get("resample", "lanczos"))
                    if str(opts.get("resample", "lanczos")) in pc.RESAMPLE_FILTERS
                    else "lanczos",
                    sharpen_after_resize=bool(opts.get("sharpen", True)),
                    jpeg_subsampling=str(opts.get("subsampling", "4:2:0")),
                    png_quantize=bool(opts.get("pngQuantize", False)),
                    strip_exif=bool(opts.get("stripExif", True)),
                    keep_icc=True,
                    skip_if_larger=True,
                )
                outcome = pc.compress_one(src_path, out_dir, cfg)
            elif mode == "video":
                cfg = vc.VideoConfig(
                    codec=str(opts.get("codec", "h265")),
                    crf=int(opts.get("crf", 27)),
                    target_mb=float(opts.get("targetMb", 0)),
                    preset=str(opts.get("preset", "medium")),
                    max_height=int(opts.get("maxHeight", 0)),
                    max_fps=int(opts.get("maxFps", 0)),
                    audio_mode=str(opts.get("audioMode", "aac128")),
                    trim_start=str(opts.get("trimStart", "")),
                    trim_end=str(opts.get("trimEnd", "")),
                    extract_audio=bool(opts.get("extractAudio", False)),
                    audio_format=str(opts.get("audioFormat", "mp3")),
                    strip_metadata=True,
                    faststart=True,
                    skip_if_larger=True,
                )
                outcome = vc.compress_one(src_path, out_dir, cfg, show_progress=False, progress_cb=on_progress)
            else:
                raise ValueError(f"Mode tidak dikenal: {mode}")

            if outcome.status == "error" or outcome.dst is None:
                raise RuntimeError(outcome.detail or "Kompresi gagal")

            out_name = outcome.dst.name
            output_key = f"jobs/{job_id}/output/{out_name}"
            s3.upload_file(str(outcome.dst), bucket, output_key)

            # Keep cached input under jobs/cache/ so same file reuses cache until 1-day B2 lifecycle expires.
            if not input_key.startswith("jobs/cache/"):
                try:
                    s3.delete_object(Bucket=bucket, Key=input_key)
                except Exception:
                    pass

            job.update(
                {
                    "status": "done",
                    "progress": 100,
                    "outputKey": output_key,
                    "outputName": out_name,
                    "origBytes": outcome.orig_bytes,
                    "newBytes": outcome.new_bytes,
                    "detail": outcome.detail,
                    "isAudioOnly": bool(mode == "video" and opts.get("extractAudio")),
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
