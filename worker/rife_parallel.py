#!/usr/bin/env python3
"""Parallel AI RIFE 60fps orchestrator (prepare -> up to 8 parallel chunk workers -> lossless merge)."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_compress as vc

MAX_WORKERS = 8


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


def read_status(s3, bucket: str, job_id: str) -> dict:
    raw = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/status.json")["Body"].read()
    return json.loads(raw.decode("utf-8"))


def fail_job(s3, bucket: str, job_id: str, msg: str) -> None:
    try:
        job = read_status(s3, bucket, job_id)
        job.update({"status": "error", "error": msg})
        write_status(s3, bucket, job_id, job)
    except Exception:
        pass


def count_video_frames(path: Path) -> int:
    cmd = [
        vc.FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out) if out.isdigit() else 0


def build_cfg_from_opts(opts: dict) -> vc.VideoConfig:
    return vc.VideoConfig(
        output_mode="video",
        codec=str(opts.get("codec", "h265")),
        crf=int(opts.get("crf", 27)),
        target_mb=float(opts.get("targetMb", 0)),
        preset=str(opts.get("preset", "medium")),
        max_height=int(opts.get("maxHeight", 0)),
        max_fps=0,
        smooth_60fps=True,
        smooth_60fps_quality=str(opts.get("smooth60Quality", "fast")),
        audio_mode=str(opts.get("audioMode", "aac96")),
        trim_start=str(opts.get("trimStart", "")),
        trim_end=str(opts.get("trimEnd", "")),
        crop_ratio=str(opts.get("cropRatio", "orig")),
        rotate_deg=int(opts.get("rotateDeg", 0)),
        speed=float(opts.get("speed", 1.0)),
        extract_audio=False,
        audio_format="mp3",
        strip_metadata=True,
        faststart=True,
        skip_if_larger=False,
    )


def cmd_prepare(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    job = read_status(s3, bucket, job_id)
    job.update({"status": "processing", "progress": 5, "speed": "Menyiapkan paralel..."})
    write_status(s3, bucket, job_id, job)

    try:
        opts = job.get("options") or {}
        cfg = build_cfg_from_opts(opts)
        items = job.get("items") or [{"inputKey": job["inputKey"], "filename": job.get("filename") or Path(job["inputKey"]).name}]
        item = items[0]
        in_key = item["inputKey"]
        orig_name = Path(item.get("filename") or in_key).name

        with tempfile.TemporaryDirectory(prefix=f"rife_prep_{job_id}_") as td:
            work = Path(td)
            src_path = work / orig_name
            s3.download_file(bucket, in_key, str(src_path))
            orig_bytes = src_path.stat().st_size
            info = vc.probe_video(src_path)

            t_start = vc.parse_timestamp(cfg.trim_start)
            t_end = vc.parse_timestamp(cfg.trim_end)
            trim_args: list[str] = []
            if t_start is not None and t_start > 0:
                trim_args += ["-ss", f"{t_start:.3f}"]
            if t_end is not None and t_end > 0:
                trim_args += ["-t", f"{max(0.1, t_end - (t_start or 0.0)):.3f}"]

            spd = max(0.5, min(2.0, float(cfg.speed or 1.0)))
            vf_list = vc.build_base_vf_list(info, cfg)

            # 1. Normalize input into intra-only high-quality CFR intermediate so frame slicing is 100% exact
            norm_path = work / "norm.mp4"
            norm_cmd = [vc.FFMPEG_BIN, "-hide_banner", "-y", "-v", "error", *trim_args, "-i", str(src_path)]
            if vf_list:
                norm_cmd += ["-vf", ",".join(vf_list)]
            norm_cmd += [
                "-an",
                "-c:v", "libx264",
                "-crf", "14",
                "-preset", "ultrafast",
                "-g", "1",
                "-pix_fmt", "yuv420p",
                str(norm_path),
            ]
            subprocess.run(norm_cmd, check=True)

            norm_info = vc.probe_video(norm_path)
            total_frames = count_video_frames(norm_path)
            eff_fps = norm_info.fps if norm_info.fps > 1.0 else (info.fps * spd if info.fps > 1.0 else 30.0)
            dur = norm_info.duration if norm_info.duration > 0 else (total_frames / max(eff_fps, 1.0))

            # 2. Extract audio once if present & not muted
            audio_key = ""
            if info.has_audio and cfg.audio_mode != "mute":
                audio_path = work / "audio.m4a"
                af_args = ["-af", f"atempo={spd:.4f}"] if abs(spd - 1.0) > 0.01 else []
                a_cmd = [
                    vc.FFMPEG_BIN, "-hide_banner", "-y", "-v", "error",
                    *trim_args, "-i", str(src_path), "-vn", *af_args,
                    "-c:a", "aac", "-b:a", "192k", str(audio_path),
                ]
                subprocess.run(a_cmd, check=True)
                audio_key = f"jobs/{job_id}/chunks/audio.m4a"
                s3.upload_file(str(audio_path), bucket, audio_key)

            # 3. Decide optimal parallel chunk count (1 chunk per ~3s, capped at 8 parallel runners)
            already_60 = eff_fps >= 55.0
            if already_60 or total_frames < 60:
                num_chunks = 1
            else:
                num_chunks = min(MAX_WORKERS, max(2, int(math.ceil(dur / 3.0))))
                num_chunks = min(num_chunks, max(1, total_frames // 20))

            # 4. Split norm.mp4 into exact overlapping frame slices (1-frame overlap at boundaries)
            chunks_meta = []
            for k in range(num_chunks):
                f_start = (k * (total_frames - 1)) // num_chunks
                f_end = ((k + 1) * (total_frames - 1)) // num_chunks  # inclusive
                if k == num_chunks - 1:
                    f_end = total_frames - 1
                c_path = work / f"in_{k:02d}.mp4"
                c_cmd = [
                    vc.FFMPEG_BIN, "-hide_banner", "-y", "-v", "error",
                    "-i", str(norm_path),
                    "-vf", f"select='between(n\\,{f_start}\\,{f_end})',setpts=PTS-STARTPTS",
                    "-vsync", "0",
                    "-c:v", "libx264", "-crf", "14", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(c_path),
                ]
                subprocess.run(c_cmd, check=True)
                c_key = f"jobs/{job_id}/chunks/in_{k:02d}.mp4"
                s3.upload_file(str(c_path), bucket, c_key)
                chunks_meta.append({
                    "idx": k,
                    "inputKey": c_key,
                    "outputKey": f"jobs/{job_id}/chunks/out_{k:02d}.mp4",
                    "frames": f_end - f_start + 1,
                    "isLast": k == num_chunks - 1,
                })

            plan = {
                "jobId": job_id,
                "origName": orig_name,
                "origBytes": orig_bytes,
                "width": norm_info.width,
                "height": norm_info.height,
                "srcFps": eff_fps,
                "duration": dur,
                "already60": already_60,
                "audioKey": audio_key,
                "numChunks": num_chunks,
                "chunks": chunks_meta,
            }
            s3.put_object(
                Bucket=bucket,
                Key=f"jobs/{job_id}/chunks/plan.json",
                Body=json.dumps(plan).encode("utf-8"),
                ContentType="application/json",
            )

            job.update({
                "progress": 14,
                "speed": f"{num_chunks} server paralel",
            })
            write_status(s3, bucket, job_id, job)

            gh_out = os.environ.get("GITHUB_OUTPUT")
            chunk_ids = list(range(num_chunks))
            if gh_out:
                with open(gh_out, "a", encoding="utf-8") as fh:
                    fh.write(f"chunks={json.dumps(chunk_ids)}\n")
            print(f"Prepared {num_chunks} parallel chunks ({total_frames} frames @ {eff_fps:.2f}fps, {norm_info.width}x{norm_info.height})")
    except Exception as exc:
        fail_job(s3, bucket, job_id, str(exc))
        raise


def cmd_chunk(job_id: str, chunk_idx: int) -> None:
    import rife_smooth

    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    try:
        job = read_status(s3, bucket, job_id)
        opts = job.get("options") or {}
        rife_quality = str(opts.get("smooth60Quality", "fast"))

        plan_raw = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/chunks/plan.json")["Body"].read()
        plan = json.loads(plan_raw.decode("utf-8"))
        cmeta = plan["chunks"][chunk_idx]

        with tempfile.TemporaryDirectory(prefix=f"rife_c{chunk_idx}_{job_id}_") as td:
            work = Path(td)
            in_path = work / f"in_{chunk_idx:02d}.mp4"
            out_path = work / f"out_{chunk_idx:02d}.mp4"
            s3.download_file(bucket, cmeta["inputKey"], str(in_path))

            if plan.get("already60"):
                out_path = in_path
            else:
                c_dur = cmeta["frames"] / max(plan["srcFps"], 1.0)
                last_push = 0.0

                def chunk_cb(pct: float, label: str) -> None:
                    nonlocal last_push
                    # Only chunk 0 pushes granular progress during execution to avoid S3 write races
                    if chunk_idx != 0:
                        return
                    import time
                    now = time.monotonic()
                    if now - last_push < 4.0:
                        return
                    last_push = now
                    try:
                        j = read_status(s3, bucket, job_id)
                        overall = round(15.0 + pct * 0.72, 1)
                        if overall > float(j.get("progress") or 0):
                            j["progress"] = overall
                            j["speed"] = f"{label} ({plan['numChunks']}x paralel)"
                            write_status(s3, bucket, job_id, j)
                    except Exception:
                        pass

                rife_smooth.interpolate_video_60fps(
                    src_path=in_path,
                    dst_path=out_path,
                    width=plan["width"],
                    height=plan["height"],
                    src_fps=plan["srcFps"],
                    duration=c_dur,
                    pre_vf=[],
                    trim_args=[],
                    codec_args=["-c:v", "libx264", "-crf", "15", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                    audio_args=["-an"],
                    extra_mux_args=[],
                    ffmpeg_bin=vc.FFMPEG_BIN,
                    progress_cb=chunk_cb,
                    rife_quality=rife_quality,
                    drop_last_frame=not cmeta["isLast"],
                )

            s3.upload_file(str(out_path), bucket, cmeta["outputKey"])
            print(f"Chunk {chunk_idx}/{plan['numChunks']} uploaded to {cmeta['outputKey']}")
    except Exception as exc:
        fail_job(s3, bucket, job_id, f"Chunk {chunk_idx} gagal: {exc}")
        raise


def cmd_merge(job_id: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    s3 = make_s3_client()
    try:
        job = read_status(s3, bucket, job_id)
        opts = job.get("options") or {}
        cfg = build_cfg_from_opts(opts)

        plan_raw = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/chunks/plan.json")["Body"].read()
        plan = json.loads(plan_raw.decode("utf-8"))

        job.update({"progress": 90, "speed": "Menggabungkan hasil 60fps..."})
        write_status(s3, bucket, job_id, job)

        with tempfile.TemporaryDirectory(prefix=f"rife_merge_{job_id}_") as td:
            work = Path(td)
            concat_file = work / "list.txt"
            lines = []
            for cmeta in plan["chunks"]:
                p = work / f"out_{cmeta['idx']:02d}.mp4"
                s3.download_file(bucket, cmeta["outputKey"], str(p))
                lines.append(f"file '{p.as_posix()}'\n")
            concat_file.write_text("".join(lines), encoding="utf-8")

            audio_path = None
            if plan.get("audioKey"):
                audio_path = work / "audio.m4a"
                s3.download_file(bucket, plan["audioKey"], str(audio_path))

            ext = vc.codec_extension(cfg.codec)
            orig_stem = Path(plan["origName"]).stem
            out_name = f"{orig_stem}{ext}"
            out_path = work / out_name

            dur = float(plan.get("duration") or 1.0)
            audio_kbps = 96 if cfg.audio_mode == "aac96" else 128
            if cfg.target_mb > 0 and dur > 0:
                total_kbps = int((cfg.target_mb * 8192 * 0.97) / dur)
                v_args = vc.build_video_codec_args(cfg, bitrate_k=max(250, total_kbps - audio_kbps))
            else:
                v_args = vc.build_video_codec_args(cfg)

            mux_args = ["-movflags", "+faststart"] if ext == ".mp4" else []
            meta_args = ["-map_metadata", "-1"]

            cmd = [vc.FFMPEG_BIN, "-hide_banner", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file)]
            if audio_path and audio_path.exists():
                cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
                if cfg.codec == "vp9" and "libopus" in vc.AVAILABLE_ENCODERS:
                    cmd += ["-c:a", "libopus", "-b:a", f"{audio_kbps}k"]
                else:
                    cmd += ["-c:a", "aac", "-b:a", f"{audio_kbps}k"]
                cmd += ["-shortest"]
            else:
                cmd += ["-an"]

            cmd += [*v_args, *meta_args, *mux_args, str(out_path)]
            subprocess.run(cmd, check=True)

            output_key = f"jobs/{job_id}/output/{out_name}"
            mime = "video/webm" if ext == ".webm" else "video/mp4"
            s3.upload_file(str(out_path), bucket, output_key, ExtraArgs={"ContentType": mime})
            new_bytes = out_path.stat().st_size

            # Cleanup chunk artifacts
            for cmeta in plan["chunks"]:
                for k in (cmeta["inputKey"], cmeta["outputKey"]):
                    try:
                        s3.delete_object(Bucket=bucket, Key=k)
                    except Exception:
                        pass
            for extra_k in (plan.get("audioKey"), f"jobs/{job_id}/chunks/plan.json"):
                if extra_k:
                    try:
                        s3.delete_object(Bucket=bucket, Key=extra_k)
                    except Exception:
                        pass

            rife_label = "AI RIFE Ensemble 60fps" if cfg.smooth_60fps_quality == "max" else "AI RIFE 60fps"
            detail = f"{cfg.codec.upper()} HD + {rife_label} ({plan['numChunks']}x paralel)"
            job.update(
                {
                    "status": "done",
                    "progress": 100,
                    "speed": "",
                    "isBatch": False,
                    "outputKey": output_key,
                    "outputName": out_name,
                    "origBytes": plan["origBytes"],
                    "newBytes": new_bytes,
                    "detail": detail,
                    "isAudioOnly": False,
                    "isAnimatedImage": False,
                }
            )
            write_status(s3, bucket, job_id, job)
            print(f"Merged {plan['numChunks']} chunks -> {out_name} ({new_bytes} B)")
    except Exception as exc:
        fail_job(s3, bucket, job_id, f"Gagal menggabungkan video: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--job-id", required=True)
    p2 = sub.add_parser("chunk")
    p2.add_argument("--job-id", required=True)
    p2.add_argument("--chunk-idx", type=int, required=True)
    p3 = sub.add_parser("merge")
    p3.add_argument("--job-id", required=True)

    args = parser.parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args.job_id)
    elif args.cmd == "chunk":
        cmd_chunk(args.job_id, args.chunk_idx)
    elif args.cmd == "merge":
        cmd_merge(args.job_id)


if __name__ == "__main__":
    main()
