#!/usr/bin/env python3
"""Media compression worker: single & multi-file batch with auto-ZIP, live progress, trim, crop, rotate, speed, GIF/WebP."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import time
import zipfile

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


def process_item(src_path: Path, out_dir: Path, mode: str, opts: dict, progress_cb):
    if mode == "photo":
        resample = str(opts.get("resample", "lanczos"))
        cfg = pc.CompressConfig(
            output_format=str(opts.get("format", "keep")),
            quality=int(opts.get("quality", 85)),
            target_kb=int(opts.get("targetKb", 0)),
            max_side=int(opts.get("maxSide", 0)),
            crop_ratio=str(opts.get("cropRatio", "orig")),
            rotate_deg=int(opts.get("rotateDeg", 0)),
            resample=resample if resample in pc.RESAMPLE_FILTERS else "lanczos",
            sharpen_after_resize=bool(opts.get("sharpen", True)),
            jpeg_subsampling=str(opts.get("subsampling", "4:2:0")),
            png_quantize=bool(opts.get("pngQuantize", False)),
            strip_exif=bool(opts.get("stripExif", True)),
            keep_icc=True,
            skip_if_larger=True,
        )
        return pc.compress_one(src_path, out_dir, cfg)

    if mode == "video":
        output_mode = str(opts.get("outputMode", "audio" if opts.get("extractAudio") else "video"))
        cfg = vc.VideoConfig(
            output_mode=output_mode,
            codec=str(opts.get("codec", "h265")),
            crf=int(opts.get("crf", 27)),
            target_mb=float(opts.get("targetMb", 0)),
            preset=str(opts.get("preset", "medium")),
            max_height=int(opts.get("maxHeight", 0)),
            max_fps=int(opts.get("maxFps", 0)),
            audio_mode=str(opts.get("audioMode", "aac96")),
            trim_start=str(opts.get("trimStart", "")),
            trim_end=str(opts.get("trimEnd", "")),
            crop_ratio=str(opts.get("cropRatio", "orig")),
            rotate_deg=int(opts.get("rotateDeg", 0)),
            speed=float(opts.get("speed", 1.0)),
            extract_audio=(output_mode == "audio"),
            audio_format=str(opts.get("audioFormat", "mp3")),
            strip_metadata=True,
            faststart=True,
            skip_if_larger=True,
        )
        return vc.compress_one(src_path, out_dir, cfg, show_progress=False, progress_cb=progress_cb)

    raise ValueError(f"Mode tidak dikenal: {mode}")


def run_job(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    status_key = f"jobs/{job_id}/status.json"

    raw = s3.get_object(Bucket=bucket, Key=status_key)["Body"].read()
    job = json.loads(raw.decode("utf-8"))
    job.update({"status": "processing", "progress": 2, "speed": ""})
    write_status(s3, bucket, job_id, job)

    last_push = 0.0

    def push_progress(pct: float, speed: str = "") -> None:
        nonlocal last_push
        now = time.monotonic()
        if now - last_push < 2.5:
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
        items = job.get("items") or [{"inputKey": job["inputKey"], "filename": job.get("filename") or Path(job["inputKey"]).name}]
        total_items = len(items)

        with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as td:
            work = Path(td)
            out_dir = work / "out"
            out_dir.mkdir()

            outcomes = []
            for idx, item in enumerate(items):
                in_key = item["inputKey"]
                orig_name = Path(item.get("filename") or in_key).name
                item_in_dir = work / f"in_{idx}"
                item_in_dir.mkdir()
                src_path = item_in_dir / orig_name
                s3.download_file(bucket, in_key, str(src_path))

                item_out_dir = out_dir / f"item_{idx}"
                item_out_dir.mkdir()

                def item_cb(p: float, spd: str, i=idx):
                    overall = ((i + p / 100.0) / total_items) * 100.0
                    push_progress(overall, spd)

                outcome = process_item(src_path, item_out_dir, mode, opts, item_cb)
                if outcome.status == "error" or outcome.dst is None:
                    raise RuntimeError(f"{orig_name}: {outcome.detail or 'gagal diproses'}")
                outcomes.append(outcome)
                push_progress(((idx + 1) / total_items) * 95.0, "")

                if not in_key.startswith("jobs/cache/"):
                    try:
                        s3.delete_object(Bucket=bucket, Key=in_key)
                    except Exception:
                        pass

            if total_items == 1:
                oc = outcomes[0]
                out_name = oc.dst.name
                output_key = f"jobs/{job_id}/output/{out_name}"
                s3.upload_file(str(oc.dst), bucket, output_key)
                out_mode = str(opts.get("outputMode", "audio" if opts.get("extractAudio") else "video"))
                job.update(
                    {
                        "status": "done",
                        "progress": 100,
                        "isBatch": False,
                        "outputKey": output_key,
                        "outputName": out_name,
                        "origBytes": oc.orig_bytes,
                        "newBytes": oc.new_bytes,
                        "detail": oc.detail,
                        "isAudioOnly": bool(mode == "video" and out_mode == "audio"),
                        "isAnimatedImage": bool(mode == "video" and out_mode in ("gif", "webp_anim")),
                    }
                )
            else:
                zip_name = f"kompres_media_{total_items}_file.zip"
                zip_path = work / zip_name
                used_names: dict[str, int] = {}
                batch_rows = []
                total_orig = 0
                total_new = 0
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                    for oc in outcomes:
                        arc = oc.dst.name
                        if arc in used_names:
                            used_names[arc] += 1
                            arc = f"{oc.dst.stem}_{used_names[arc]}{oc.dst.suffix}"
                        else:
                            used_names[arc] = 1
                        zf.write(oc.dst, arcname=arc)
                        total_orig += oc.orig_bytes
                        total_new += oc.new_bytes
                        batch_rows.append(
                            {
                                "name": arc,
                                "origBytes": oc.orig_bytes,
                                "newBytes": oc.new_bytes,
                                "detail": oc.detail,
                            }
                        )
                output_key = f"jobs/{job_id}/output/{zip_name}"
                s3.upload_file(str(zip_path), bucket, output_key)
                job.update(
                    {
                        "status": "done",
                        "progress": 100,
                        "isBatch": True,
                        "outputKey": output_key,
                        "outputName": zip_name,
                        "origBytes": total_orig,
                        "newBytes": total_new,
                        "detail": f"{total_items} file dikemas ke ZIP",
                        "batchItems": batch_rows,
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
