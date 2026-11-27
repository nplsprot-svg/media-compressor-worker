#!/usr/bin/env python3
"""HD video processor: CRF/2-pass compression, trim, crop, rotate, speed, audio extract, GIF & animated WebP."""

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
    for name in ("libx264", "libx265", "libvpx-vp9", "libwebp", "aac", "libopus", "libmp3lame"):
        if f" {name} " in proc.stdout:
            found.add(name)
    return found


AVAILABLE_ENCODERS = detect_encoders()


@dataclass
class VideoConfig:
    input_path: str = ""
    output_dir: str = ""
    overwrite: bool = False
    output_mode: str = "video"  # video | audio | gif | webp_anim
    codec: str = "h265"
    crf: int = 27
    target_mb: float = 0.0
    preset: str = "medium"
    max_height: int = 0
    max_fps: int = 0
    smooth_60fps: bool = False
    audio_mode: str = "aac96"
    trim_start: str = ""
    trim_end: str = ""
    crop_ratio: str = "orig"  # orig | 9:16 | 1:1 | 16:9 | 4:5
    rotate_deg: int = 0       # 0 | 90 | 180 | 270 clockwise
    speed: float = 1.0        # 0.5 .. 2.0
    extract_audio: bool = False
    audio_format: str = "mp3"
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


def build_base_vf_list(info: VideoInfo, cfg: VideoConfig) -> list[str]:
    filters: list[str] = []
    deg = int(cfg.rotate_deg or 0) % 360
    if deg == 90:
        filters.append("transpose=1")
    elif deg == 180:
        filters.append("transpose=1,transpose=1")
    elif deg == 270:
        filters.append("transpose=2")

    ratio_exprs = {
        "9:16": ("9/16", "16/9"),
        "1:1": ("1", "1"),
        "16:9": ("16/9", "9/16"),
        "4:5": ("4/5", "5/4"),
    }
    if cfg.crop_ratio in ratio_exprs:
        r_wh, r_hw = ratio_exprs[cfg.crop_ratio]
        filters.append(
            f"crop='trunc(min(iw,ih*{r_wh})/2)*2':'trunc(min(ih,iw*{r_hw})/2)*2'"
        )

    if cfg.max_height > 0:
        h = cfg.max_height
        filters.append(
            f"scale='if(gt(iw,ih),-2,min(iw,{h}))':'if(gt(iw,ih),min(ih,{h}),-2)':flags=lanczos"
        )

    spd = float(cfg.speed or 1.0)
    if abs(spd - 1.0) > 0.01:
        spd = max(0.5, min(2.0, spd))
        filters.append(f"setpts=PTS/{spd:.4f}")

    if cfg.smooth_60fps:
        # Motion-compensated optical flow interpolation to smooth 60fps (skips if already ~60fps).
        if info.fps <= 0 or info.fps < 55.0:
            filters.append("minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    elif cfg.max_fps > 0 and (info.fps <= 0 or info.fps > cfg.max_fps + 0.5):
        filters.append(f"fps={cfg.max_fps}")

    return filters


def build_audio_args(info: VideoInfo, cfg: VideoConfig) -> tuple[list[str], int]:
    if not info.has_audio or cfg.audio_mode == "mute":
        return ["-an"], 0
    spd = max(0.5, min(2.0, float(cfg.speed or 1.0)))
    speed_changed = abs(spd - 1.0) > 0.01

    if cfg.audio_mode == "copy" and cfg.codec != "vp9" and not speed_changed:
        return ["-c:a", "copy"], 128

    bitrate_k = 96 if cfg.audio_mode == "aac96" else 128
    af_args = ["-af", f"atempo={spd:.4f}"] if speed_changed else []
    if cfg.codec == "vp9" and "libopus" in AVAILABLE_ENCODERS:
        return [*af_args, "-c:a", "libopus", "-b:a", f"{bitrate_k}k"], bitrate_k
    return [*af_args, "-c:a", "aac", "-b:a", f"{bitrate_k}k"], bitrate_k


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
            raise ValueError("Waktu selesai potong harus lebih besar dari waktu mulai")

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

        spd = max(0.5, min(2.0, float(cfg.speed or 1.0)))
        if eff_duration > 0 and abs(spd - 1.0) > 0.01:
            eff_duration = eff_duration / spd

        mode = "audio" if cfg.extract_audio else (cfg.output_mode or "video")
        meta_args = ["-map_metadata", "-1"] if cfg.strip_metadata else []

        # 1. Audio extraction mode
        if mode == "audio":
            if not info.has_audio:
                raise RuntimeError("Video ini tidak memiliki trek audio")
            a_fmt = "m4a" if cfg.audio_format == "m4a" else "mp3"
            if a_fmt == "mp3" and "libmp3lame" in AVAILABLE_ENCODERS:
                ext, a_enc = ".mp3", ["-c:a", "libmp3lame", "-q:a", "2"]
            else:
                ext, a_enc = ".m4a", ["-c:a", "aac", "-b:a", "192k"]
            af_args = ["-af", f"atempo={spd:.4f}"] if abs(spd - 1.0) > 0.01 else []
            out_name = f"{src.stem}{ext}"
            dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(f".{dst.stem}.tmp{ext}")
            cmd = [*trim_args, "-i", str(src), "-vn", *af_args, *a_enc, *meta_args, str(tmp_dst)]
            run_ffmpeg_with_progress(cmd, eff_duration, src.name, show_progress, progress_cb)
            new_bytes = tmp_dst.stat().st_size
            tmp_dst.replace(dst)
            return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", f"Ekstrak audio ({ext.lstrip('.').upper()})")

        # 2. Animated GIF mode
        if mode == "gif":
            base_vf = build_base_vf_list(info, VideoConfig(**{**cfg.__dict__, "max_height": 0, "max_fps": 0}))
            gif_scale = "fps=12,scale='min(iw,480)':-2:flags=lanczos"
            chain = ",".join([*base_vf, gif_scale])
            vf_complex = f"{chain},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3"
            out_name = f"{src.stem}.gif"
            dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(f".{dst.stem}.tmp.gif")
            cmd = [*trim_args, "-i", str(src), "-vf", vf_complex, "-an", "-loop", "0", str(tmp_dst)]
            run_ffmpeg_with_progress(cmd, eff_duration, src.name, show_progress, progress_cb)
            new_bytes = tmp_dst.stat().st_size
            tmp_dst.replace(dst)
            return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", "GIF animasi (12 fps)")

        # 3. Animated WebP sticker/clip mode
        if mode == "webp_anim":
            if "libwebp" not in AVAILABLE_ENCODERS:
                raise RuntimeError("Encoder WebP animasi tidak tersedia")
            base_vf = build_base_vf_list(info, VideoConfig(**{**cfg.__dict__, "max_height": 0, "max_fps": 0}))
            webp_vf = ",".join([*base_vf, "fps=12,scale='min(iw,512)':-2:flags=lanczos"])
            out_name = f"{src.stem}.webp"
            dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_dst = dst.with_name(f".{dst.stem}.tmp.webp")
            cmd = [
                *trim_args, "-i", str(src),
                "-vf", webp_vf,
                "-c:v", "libwebp", "-lossless", "0", "-compression_level", "4", "-q:v", "75",
                "-loop", "0", "-an",
                str(tmp_dst),
            ]
            run_ffmpeg_with_progress(cmd, eff_duration, src.name, show_progress, progress_cb)
            new_bytes = tmp_dst.stat().st_size
            tmp_dst.replace(dst)
            return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", "WebP animasi (12 fps)")

        # 4. Standard HD video compression
        vf_list = build_base_vf_list(info, cfg)
        vf = ",".join(vf_list) if vf_list else None
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
                    *common_in, *vf_args, *v_args, *p2_extra,
                    *audio_args, *meta_args, *mux_args, str(tmp_dst),
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
        if abs(spd - 1.0) > 0.01:
            note += f" {spd:g}x"
        if cfg.smooth_60fps:
            note += " + Smooth 60fps"
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
