#!/usr/bin/env python3
"""HD video compressor powered by FFmpeg with live progress callback, trim, and audio extract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".3gp", ".ts"}
FFMPEG_BIN = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")


def detect_encoders() -> set[str]:
    if not FFMPEG_BIN:
        return set()
    proc = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    found: set[str] = set()
    for name in ("libx264", "libx265", "libvpx-vp9", "libsvtav1", "aac", "libopus", "libmp3lame"):
        if f" {name} " in proc.stdout:
            found.add(name)
    return found


AVAILABLE_ENCODERS = detect_encoders()


@dataclass
class VideoConfig:
    input_path: str = ""
    output_dir: str = ""
    overwrite: bool = False
    codec: str = "h264"  # h264 | h265 | vp9
    crf: int = 23
    target_mb: float = 0.0
    preset: str = "medium"
    max_height: int = 0
    max_fps: int = 0
    audio_mode: str = "aac128"  # aac128 | aac96 | copy | mute
    trim_start: str = ""  # e.g. "00:05" or "5"
    trim_end: str = ""    # e.g. "00:30" or "30"
    extract_audio: bool = False
    audio_format: str = "mp3"  # mp3 | m4a
    strip_metadata: bool = True
    faststart: bool = True
    recursive: bool = False
    skip_if_larger: bool = True


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    video_codec: str
    has_audio: bool


@dataclass
class VideoOutcome:
    src: Path
    dst: Path | None
    orig_bytes: int
    new_bytes: int
    status: str  # ok | skipped | error
    detail: str = ""


def parse_timestamp(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Format waktu trim tidak valid: {raw}")
    if len(nums) == 1:
        return max(0.0, nums[0])
    if len(nums) == 2:
        return max(0.0, nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return max(0.0, nums[0] * 3600 + nums[1] * 60 + nums[2])
    raise ValueError(f"Format waktu trim tidak valid: {raw}")


def codec_encoder_name(codec: str) -> str:
    return {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}[codec]


def codec_extension(codec: str) -> str:
    return ".webm" if codec == "vp9" else ".mp4"


def parse_fps(raw: str) -> float:
    if not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def probe_video(path: Path) -> VideoInfo:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Gagal membaca informasi video")

    meta = json.loads(proc.stdout)
    streams = meta.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise RuntimeError("Tidak ada stream video di file ini")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(meta.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)
    fps = parse_fps(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate") or "")
    return VideoInfo(
        width=int(v_stream.get("width", 0)),
        height=int(v_stream.get("height", 0)),
        duration=duration,
        fps=fps,
        video_codec=str(v_stream.get("codec_name", "")),
        has_audio=has_audio,
    )


def build_vf_chain(info: VideoInfo, cfg: VideoConfig) -> str | None:
    filters: list[str] = []
    if cfg.max_height > 0:
        h = cfg.max_height
        filters.append(
            f"scale='if(gt(iw,ih),-2,min(iw,{h}))':'if(gt(iw,ih),min(ih,{h}),-2)':flags=lanczos"
        )
    if cfg.max_fps > 0 and (info.fps <= 0 or info.fps > cfg.max_fps + 0.5):
        filters.append(f"fps={cfg.max_fps}")
    return ",".join(filters) if filters else None


def build_audio_args(info: VideoInfo, cfg: VideoConfig) -> tuple[list[str], int]:
    if not info.has_audio or cfg.audio_mode == "mute":
        return ["-an"], 0
    if cfg.audio_mode == "copy" and cfg.codec != "vp9":
        return ["-c:a", "copy"], 128
    bitrate_k = 96 if cfg.audio_mode == "aac96" else 128
    if cfg.codec == "vp9" and "libopus" in AVAILABLE_ENCODERS:
        return ["-c:a", "libopus", "-b:a", f"{bitrate_k}k"], bitrate_k
    return ["-c:a", "aac", "-b:a", f"{bitrate_k}k"], bitrate_k


def build_video_codec_args(cfg: VideoConfig, bitrate_k: int | None = None) -> list[str]:
    enc = codec_encoder_name(cfg.codec)
    args = ["-c:v", enc, "-pix_fmt", "yuv420p"]

    if cfg.codec == "h264":
        args += ["-preset", cfg.preset]
        args += (["-b:v", f"{bitrate_k}k"] if bitrate_k is not None else ["-crf", str(cfg.crf)])
    elif cfg.codec == "h265":
        args += ["-preset", cfg.preset, "-tag:v", "hvc1"]
        args += (["-b:v", f"{bitrate_k}k"] if bitrate_k is not None else ["-crf", str(cfg.crf)])
    elif cfg.codec == "vp9":
        cpu_used = {"ultrafast": "5", "veryfast": "4", "fast": "3", "medium": "2", "slow": "1"}
        args += ["-cpu-used", cpu_used.get(cfg.preset, "2"), "-row-mt", "1"]
        args += (["-b:v", f"{bitrate_k}k"] if bitrate_k is not None else ["-crf", str(cfg.crf), "-b:v", "0"])

    return args


def run_ffmpeg_with_progress(
    cmd: list[str],
    duration: float,
    label: str,
    show_progress: bool,
    progress_cb: Callable[[float, str], None] | None = None,
) -> None:
    full_cmd = [FFMPEG_BIN, "-hide_banner", "-y", "-nostats", "-progress", "pipe:1", *cmd]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stderr is not None

    speed = ""
    last_print = 0.0
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("speed="):
            speed = line.split("=", 1)[1].strip()
        elif line.startswith("out_time_us=") and duration > 0:
            raw_us = line.split("=", 1)[1].strip()
            if not raw_us.isdigit():
                continue
            pct = min(99.5, (int(raw_us) / 1_000_000) / duration * 100)
            if progress_cb:
                progress_cb(pct, speed)
            if show_progress:
                now = time.monotonic()
                if now - last_print >= 0.35:
                    last_print = now
                    sys.stdout.write(f"\r  {label}: {pct:5.1f}% (speed {speed or '?'})   ")
                    sys.stdout.flush()

    stderr_text = proc.stderr.read()
    proc.wait()
    if show_progress and duration > 0:
        sys.stdout.write("\r" + " " * 64 + "\r")
        sys.stdout.flush()
    if proc.returncode != 0:
        tail = "\n".join(stderr_text.strip().splitlines()[-4:])
        raise RuntimeError(tail or f"Encoder gagal (exit {proc.returncode})")


def compress_one(
    src: Path,
    dst_dir: Path | None,
    cfg: VideoConfig,
    show_progress: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> VideoOutcome:
    orig_bytes = src.stat().st_size
    try:
        info = probe_video(src)
        t_start = parse_timestamp(cfg.trim_start)
        t_end = parse_timestamp(cfg.trim_end)
        if t_start is not None and t_end is not None and t_end <= t_start:
            raise ValueError("Waktu selesai trim harus lebih besar dari waktu mulai")

        trim_args: list[str] = []
        eff_duration = info.duration
        if t_start is not None and t_start > 0:
            trim_args += ["-ss", f"{t_start:.3f}"]
        if t_end is not None and t_end > 0:
            clip_len = t_end - (t_start or 0.0)
            trim_args += ["-t", f"{clip_len:.3f}"]
            eff_duration = min(eff_duration, clip_len) if eff_duration > 0 else clip_len
        elif t_start is not None and eff_duration > t_start:
            eff_duration = eff_duration - t_start

        meta_args = ["-map_metadata", "-1"] if cfg.strip_metadata else []

        # Audio-only extraction mode
        if cfg.extract_audio:
            if not info.has_audio:
                raise RuntimeError("Video ini tidak memiliki trek audio")
            a_fmt = "m4a" if cfg.audio_format == "m4a" else "mp3"
            ext = f".{a_fmt}"
            out_name = f"{src.stem}{ext}"
            dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(f".{dst.stem}.tmp{ext}")

            if a_fmt == "mp3" and "libmp3lame" in AVAILABLE_ENCODERS:
                a_enc = ["-c:a", "libmp3lame", "-q:a", "2"]
            else:
                a_enc = ["-c:a", "aac", "-b:a", "192k"]
                if a_fmt == "mp3":
                    ext = ".m4a"
                    out_name = f"{src.stem}.m4a"
                    dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
                    tmp_dst = dst.with_name(f".{dst.stem}.tmp.m4a")

            cmd = [*trim_args, "-i", str(src), "-vn", *a_enc, *meta_args, str(tmp_dst)]
            run_ffmpeg_with_progress(cmd, eff_duration, src.name, show_progress, progress_cb)
            new_bytes = tmp_dst.stat().st_size
            tmp_dst.replace(dst)
            return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", f"Ekstrak audio ({ext.lstrip('.').upper()})")

        vf = build_vf_chain(info, cfg)
        audio_args, audio_kbps = build_audio_args(info, cfg)
        ext = codec_extension(cfg.codec)
        same_ext = src.suffix.lower() == ext
        out_name = src.name if same_ext else f"{src.stem}{ext}"
        dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(f".{dst.stem}.tmp{ext}")

        common_in = [*trim_args, "-i", str(src)]
        vf_args = ["-vf", vf] if vf else []
        mux_args = ["-movflags", "+faststart"] if (cfg.faststart and ext == ".mp4") else []

        if cfg.target_mb > 0 and eff_duration > 0:
            total_kbps = int((cfg.target_mb * 8192 * 0.97) / eff_duration)
            video_kbps = max(250, total_kbps - audio_kbps)
            with tempfile.TemporaryDirectory(prefix="vcomp_2pass_") as logdir:
                passlog = str(Path(logdir) / "ffmpeg2pass")
                v_args = build_video_codec_args(cfg, bitrate_k=video_kbps)
                p1_extra = (
                    ["-x265-params", f"pass=1:stats={passlog}.log"]
                    if cfg.codec == "h265"
                    else ["-pass", "1", "-passlogfile", passlog]
                )
                p2_extra = (
                    ["-x265-params", f"pass=2:stats={passlog}.log"]
                    if cfg.codec == "h265"
                    else ["-pass", "2", "-passlogfile", passlog]
                )
                pass1 = [*common_in, *vf_args, *v_args, *p1_extra, "-an", "-f", "null", os.devnull]
                pass2 = [
                    *common_in,
                    *vf_args,
                    *v_args,
                    *p2_extra,
                    *audio_args,
                    *meta_args,
                    *mux_args,
                    str(tmp_dst),
                ]
                cb1 = (lambda p, s: progress_cb(p * 0.45, s)) if progress_cb else None
                cb2 = (lambda p, s: progress_cb(45.0 + p * 0.55, s)) if progress_cb else None
                run_ffmpeg_with_progress(pass1, eff_duration, f"{src.name} [1/2]", show_progress, cb1)
                run_ffmpeg_with_progress(pass2, eff_duration, f"{src.name} [2/2]", show_progress, cb2)
            note = f"{cfg.codec.upper()} target <= {cfg.target_mb:g} MB"
        else:
            v_args = build_video_codec_args(cfg)
            cmd = [*common_in, *vf_args, *v_args, *audio_args, *meta_args, *mux_args, str(tmp_dst)]
            run_ffmpeg_with_progress(cmd, eff_duration, src.name, show_progress, progress_cb)
            note = f"{cfg.codec.upper()} HD"

        if cfg.max_height > 0:
            note += f" (maks {cfg.max_height}p)"
        if trim_args:
            note += " + dipotong"

        new_bytes = tmp_dst.stat().st_size
        if cfg.skip_if_larger and same_ext and vf is None and not trim_args and new_bytes >= orig_bytes:
            tmp_dst.unlink(missing_ok=True)
            if not cfg.overwrite:
                shutil.copy2(src, dst)
            return VideoOutcome(src, dst, orig_bytes, orig_bytes, "skipped", "File asli sudah paling kecil")

        tmp_dst.replace(dst)
        if cfg.overwrite and dst != src and src.exists():
            src.unlink()
        return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", note)
    except Exception as exc:
        return VideoOutcome(src, None, orig_bytes, orig_bytes, "error", str(exc))
