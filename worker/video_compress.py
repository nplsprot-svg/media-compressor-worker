#!/usr/bin/env python3
"""Interactive CLI HD video compressor powered by FFmpeg."""

from dataclasses import dataclass
from pathlib import Path
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
    for name in ("libx264", "libx265", "libvpx-vp9", "libsvtav1", "aac", "libopus"):
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
    crf: int = 23  # default for h264; auto-adjusted when switching codec
    target_mb: float = 0.0  # 0 = CRF mode, >0 = 2-pass target MB
    preset: str = "medium"  # ultrafast | veryfast | fast | medium | slow
    max_height: int = 0  # 0 = keep original; otherwise caps short side (1080, 720, 480)
    max_fps: int = 0  # 0 = keep original; otherwise caps fps (e.g., 30, 24)
    audio_mode: str = "aac128"  # aac128 | aac96 | copy | mute
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


def fmt_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


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
        raise RuntimeError(proc.stderr.strip() or "ffprobe gagal membaca file")

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
        # Cap the shorter side so both 16:9 landscape and 9:16 portrait stay true HD with even dims.
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
    if cfg.audio_mode == "copy":
        # Copy only when container supports original audio reliably; fallback handled if needed.
        if cfg.codec != "vp9":
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
        if bitrate_k is not None:
            args += ["-b:v", f"{bitrate_k}k"]
        else:
            args += ["-crf", str(cfg.crf)]
    elif cfg.codec == "h265":
        # hvc1 tag ensures playback compatibility on Android galleries, iOS, and browsers.
        args += ["-preset", cfg.preset, "-tag:v", "hvc1"]
        if bitrate_k is not None:
            args += ["-b:v", f"{bitrate_k}k"]
        else:
            args += ["-crf", str(cfg.crf)]
    elif cfg.codec == "vp9":
        cpu_used = {"ultrafast": "5", "veryfast": "4", "fast": "3", "medium": "2", "slow": "1"}
        args += ["-cpu-used", cpu_used.get(cfg.preset, "2"), "-row-mt", "1"]
        if bitrate_k is not None:
            args += ["-b:v", f"{bitrate_k}k"]
        else:
            args += ["-crf", str(cfg.crf), "-b:v", "0"]

    return args


def run_ffmpeg_with_progress(cmd: list[str], duration: float, label: str, show_progress: bool) -> None:
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
        elif line.startswith("out_time_us=") and show_progress and duration > 0:
            now = time.monotonic()
            if now - last_print < 0.35:
                continue
            last_print = now
            raw_us = line.split("=", 1)[1].strip()
            if raw_us.isdigit():
                pct = min(99.9, (int(raw_us) / 1_000_000) / duration * 100)
                sys.stdout.write(f"\r  {label}: {pct:5.1f}% (speed {speed or '?'})   ")
                sys.stdout.flush()

    stderr_text = proc.stderr.read()
    proc.wait()
    if show_progress and duration > 0:
        sys.stdout.write("\r" + " " * 64 + "\r")
        sys.stdout.flush()
    if proc.returncode != 0:
        tail = "\n".join(stderr_text.strip().splitlines()[-4:])
        raise RuntimeError(tail or f"ffmpeg exit code {proc.returncode}")


def compress_one(
    src: Path,
    dst_dir: Path | None,
    cfg: VideoConfig,
    show_progress: bool = True,
) -> VideoOutcome:
    orig_bytes = src.stat().st_size
    try:
        info = probe_video(src)
        vf = build_vf_chain(info, cfg)
        audio_args, audio_kbps = build_audio_args(info, cfg)
        ext = codec_extension(cfg.codec)
        same_ext = src.suffix.lower() == ext
        out_name = src.name if same_ext else f"{src.stem}{ext}"
        dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(f".{dst.stem}.tmp{ext}")

        common_in = ["-i", str(src)]
        vf_args = ["-vf", vf] if vf else []
        meta_args = ["-map_metadata", "-1"] if cfg.strip_metadata else []
        mux_args = ["-movflags", "+faststart"] if (cfg.faststart and ext == ".mp4") else []

        if cfg.target_mb > 0 and info.duration > 0:
            # Reserve ~3% container overhead so final MP4 stays under the target MB cap.
            total_kbps = int((cfg.target_mb * 8192 * 0.97) / info.duration)
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
                run_ffmpeg_with_progress(pass1, info.duration, f"{src.name} [pass 1/2]", show_progress)
                run_ffmpeg_with_progress(pass2, info.duration, f"{src.name} [pass 2/2]", show_progress)
            note = f"{cfg.codec.upper()} 2-pass ~{video_kbps}k"
        else:
            v_args = build_video_codec_args(cfg)
            cmd = [*common_in, *vf_args, *v_args, *audio_args, *meta_args, *mux_args, str(tmp_dst)]
            run_ffmpeg_with_progress(cmd, info.duration, src.name, show_progress)
            note = f"{cfg.codec.upper()} crf={cfg.crf}"

        if cfg.max_height > 0:
            note += f" <= {cfg.max_height}p"

        new_bytes = tmp_dst.stat().st_size
        if cfg.skip_if_larger and same_ext and vf is None and new_bytes >= orig_bytes:
            tmp_dst.unlink(missing_ok=True)
            if not cfg.overwrite:
                shutil.copy2(src, dst)
            return VideoOutcome(src, dst, orig_bytes, orig_bytes, "skipped", "asli lebih kecil")

        tmp_dst.replace(dst)
        if cfg.overwrite and dst != src and src.exists():
            src.unlink()
        return VideoOutcome(src, dst, orig_bytes, new_bytes, "ok", note)
    except Exception as exc:
        return VideoOutcome(src, None, orig_bytes, orig_bytes, "error", str(exc))


def collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTS else []
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "compressed_videos"
    return input_path.parent / f"{input_path.name}_compressed"


def run_batch(cfg: VideoConfig) -> None:
    if not cfg.input_path:
        print("\n[!] Input path belum diisi. Pilih menu 1 dulu.")
        return

    src_root = Path(cfg.input_path).expanduser().resolve()
    if not src_root.exists():
        print(f"\n[!] Path tidak ditemukan: {src_root}")
        return

    files = collect_videos(src_root, cfg.recursive)
    if not files:
        print(f"\n[!] Tidak ada file video yang didukung di: {src_root}")
        return

    out_root = None if cfg.overwrite else (
        Path(cfg.output_dir).expanduser().resolve() if cfg.output_dir else default_output_dir(src_root)
    )
    print(f"\nMemproses {len(files)} video...")
    print(f"Output: {out_root if out_root else 'OVERWRITE file asli'}")

    t0 = time.monotonic()
    total_orig = 0
    total_new = 0
    ok_count = 0
    skip_count = 0
    err_count = 0

    for idx, vid in enumerate(files, 1):
        dst_dir = None
        if out_root is not None:
            rel_parent = vid.parent.relative_to(src_root) if src_root.is_dir() else Path(".")
            dst_dir = out_root / rel_parent
        out = compress_one(vid, dst_dir, cfg, show_progress=True)
        total_orig += out.orig_bytes
        total_new += out.new_bytes
        if out.status == "ok":
            ok_count += 1
            pct = (1 - out.new_bytes / out.orig_bytes) * 100 if out.orig_bytes else 0.0
            print(
                f"  [{idx}/{len(files)}] OK   {vid.name}: "
                f"{fmt_size(out.orig_bytes)} -> {fmt_size(out.new_bytes)} "
                f"({pct:+.1f}%) [{out.detail}]"
            )
        elif out.status == "skipped":
            skip_count += 1
            print(f"  [{idx}/{len(files)}] SKIP {vid.name}: {out.detail}")
        else:
            err_count += 1
            print(f"  [{idx}/{len(files)}] ERR  {vid.name}: {out.detail}")

    elapsed = time.monotonic() - t0
    saved = total_orig - total_new
    saved_pct = (saved / total_orig * 100) if total_orig else 0.0
    print(
        f"\nSelesai dalam {elapsed:.1f}s | OK: {ok_count}, Skip: {skip_count}, Error: {err_count}\n"
        f"Total: {fmt_size(total_orig)} -> {fmt_size(total_new)} "
        f"(hemat {fmt_size(saved)} / {saved_pct:.1f}%)"
    )


def run_benchmark(cfg: VideoConfig) -> None:
    sample_raw = input("Path 1 video sample (kosong = pakai input saat ini): ").strip().strip("'\"")
    path = Path(sample_raw or cfg.input_path).expanduser().resolve()
    if path.is_dir():
        vids = collect_videos(path, False)
        if not vids:
            print("[!] Tidak ada video di folder tersebut.")
            return
        path = vids[0]
    if not path.is_file():
        print(f"[!] File tidak ditemukan: {path}")
        return

    info = probe_video(path)
    clip_sec = min(10.0, info.duration) if info.duration > 0 else 10.0
    orig_bytes = path.stat().st_size
    ratio = (info.duration / clip_sec) if clip_sec > 0 and info.duration > 0 else 1.0
    print(
        f"\nBenchmark {clip_sec:.1f}s pertama: {path.name} "
        f"({info.width}x{info.height}, {info.duration:.1f}s, asli {fmt_size(orig_bytes)})"
    )

    cases = []
    if "libx264" in AVAILABLE_ENCODERS:
        cases.append(("H.264 crf=23", "h264", 23))
        cases.append(("H.264 crf=26", "h264", 26))
    if "libx265" in AVAILABLE_ENCODERS:
        cases.append(("H.265 crf=27", "h265", 27))
        cases.append(("H.265 crf=30", "h265", 30))

    vf = build_vf_chain(info, cfg)
    with tempfile.TemporaryDirectory(prefix="vcomp_bench_") as td:
        for label, codec, crf in cases:
            tmp_cfg = VideoConfig(
                codec=codec,
                crf=crf,
                preset=cfg.preset,
                max_height=cfg.max_height,
                max_fps=cfg.max_fps,
                audio_mode=cfg.audio_mode,
            )
            out_file = Path(td) / f"bench_{codec}_{crf}{codec_extension(codec)}"
            audio_args, _ = build_audio_args(info, tmp_cfg)
            cmd = [
                "-t",
                f"{clip_sec:.2f}",
                "-i",
                str(path),
                *(["-vf", vf] if vf else []),
                *build_video_codec_args(tmp_cfg),
                *audio_args,
                str(out_file),
            ]
            t0 = time.monotonic()
            try:
                run_ffmpeg_with_progress(cmd, clip_sec, label, show_progress=False)
            except RuntimeError as exc:
                print(f"  {label:<14} -> GAGAL ({exc})")
                continue
            elapsed = time.monotonic() - t0
            est_total = int(out_file.stat().st_size * ratio)
            pct = (1 - est_total / orig_bytes) * 100 if orig_bytes else 0.0
            print(f"  {label:<14} -> estimasi {fmt_size(est_total):>9} ({pct:+5.1f}%)  [{elapsed:.1f}s]")


def prompt_int(label: str, default: int, min_val: int, max_val: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        print("[!] Harus angka, nilai tetap.")
        return default
    if val < min_val or val > max_val:
        print(f"[!] Rentang valid {min_val}-{max_val}, nilai tetap.")
        return default
    return val


def prompt_float(label: str, default: float, min_val: float, max_val: float) -> float:
    raw = input(f"{label} [{default:g}]: ").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        print("[!] Harus angka, nilai tetap.")
        return default
    if val < min_val or val > max_val:
        print(f"[!] Rentang valid {min_val:g}-{max_val:g}, nilai tetap.")
        return default
    return val


def menu_codec(cfg: VideoConfig) -> None:
    print("\nCodec Output:")
    print(f"  1. H.264 / MP4 (universal, cepat) {'[OK]' if 'libx264' in AVAILABLE_ENCODERS else '[TIDAK ADA]'}")
    print(f"  2. H.265 / MP4 (hemat 25-50%, HD tajam) {'[OK]' if 'libx265' in AVAILABLE_ENCODERS else '[TIDAK ADA]'}")
    print(f"  3. VP9 / WebM  (web friendly) {'[OK]' if 'libvpx-vp9' in AVAILABLE_ENCODERS else '[TIDAK ADA]'}")
    choice = input("Pilih [1-3]: ").strip()
    mapping = {"1": ("h264", 23), "2": ("h265", 27), "3": ("vp9", 31)}
    picked = mapping.get(choice)
    if not picked:
        return
    codec, default_crf = picked
    if codec_encoder_name(codec) not in AVAILABLE_ENCODERS:
        print(f"[!] Encoder {codec_encoder_name(codec)} tidak tersedia di build FFmpeg ini.")
        return
    cfg.codec = codec
    cfg.crf = default_crf


def menu_quality(cfg: VideoConfig) -> None:
    base = 27 if cfg.codec == "h265" else (31 if cfg.codec == "vp9" else 23)
    print(f"\nPreset Kualitas ({cfg.codec.upper()}):")
    print(f"  1. Ultra HD     (CRF {base - 3})")
    print(f"  2. Balanced HD  (CRF {base} - rekomendasi)")
    print(f"  3. Hemat HD     (CRF {base + 3})")
    print("  4. Custom CRF   (14-36, makin kecil = makin jernih)")
    print("  5. Target ukuran maksimal (MB) via 2-pass encoding")
    choice = input("Pilih [1-5]: ").strip()
    if choice == "1":
        cfg.crf, cfg.target_mb = base - 3, 0.0
    elif choice == "2":
        cfg.crf, cfg.target_mb = base, 0.0
    elif choice == "3":
        cfg.crf, cfg.target_mb = base + 3, 0.0
    elif choice == "4":
        cfg.crf = prompt_int("CRF (14-36)", cfg.crf, 14, 36)
        cfg.target_mb = 0.0
    elif choice == "5":
        cfg.target_mb = prompt_float("Target max MB (misal 15)", cfg.target_mb or 15.0, 1.0, 10000.0)


def menu_resolution(cfg: VideoConfig) -> None:
    print("\nResolusi & FPS (sisi pendek, anti-upscale):")
    print("  1. Pertahankan resolusi asli")
    print("  2. Max 1080p (Full HD)")
    print("  3. Max 720p  (HD ringan)")
    print("  4. Max 480p  (super hemat)")
    print(f"  5. Batas FPS (saat ini: {cfg.max_fps or 'asli'})")
    choice = input("Pilih [1-5]: ").strip()
    heights = {"1": 0, "2": 1080, "3": 720, "4": 480}
    if choice in heights:
        cfg.max_height = heights[choice]
    elif choice == "5":
        print("  0 = asli, 30 = max 30fps, 24 = max 24fps")
        cfg.max_fps = prompt_int("Max FPS", cfg.max_fps, 0, 120)


def menu_advanced(cfg: VideoConfig) -> None:
    presets = ["ultrafast", "veryfast", "fast", "medium", "slow"]
    audio_modes = ["aac128", "aac96", "copy", "mute"]
    while True:
        print("\nOpsi Lanjutan:")
        print(f"  1. Preset speed FFmpeg : {cfg.preset} (lebih lambat = lebih kecil/tajam)")
        print(f"  2. Audio               : {cfg.audio_mode}")
        print(f"  3. Strip metadata      : {'ON' if cfg.strip_metadata else 'OFF'}")
        print(f"  4. Faststart MP4       : {'ON' if cfg.faststart else 'OFF'}")
        print(f"  5. Scan subfolder      : {'ON' if cfg.recursive else 'OFF'}")
        print(f"  6. Skip jika >= asli   : {'ON' if cfg.skip_if_larger else 'OFF'}")
        print("  0. Kembali")
        choice = input("Pilih [0-6]: ").strip()
        if choice == "0" or not choice:
            return
        elif choice == "1":
            idx = (presets.index(cfg.preset) + 1) % len(presets)
            cfg.preset = presets[idx]
        elif choice == "2":
            idx = (audio_modes.index(cfg.audio_mode) + 1) % len(audio_modes)
            cfg.audio_mode = audio_modes[idx]
        elif choice == "3":
            cfg.strip_metadata = not cfg.strip_metadata
        elif choice == "4":
            cfg.faststart = not cfg.faststart
        elif choice == "5":
            cfg.recursive = not cfg.recursive
        elif choice == "6":
            cfg.skip_if_larger = not cfg.skip_if_larger


def menu_output(cfg: VideoConfig) -> None:
    print("\nTujuan Output:")
    print("  1. Folder otomatis (<folder>_compressed / compressed_videos)")
    print("  2. Folder custom")
    print("  3. Overwrite file asli (hati-hati, tidak bisa undo)")
    choice = input("Pilih [1-3]: ").strip()
    if choice == "1":
        cfg.overwrite = False
        cfg.output_dir = ""
    elif choice == "2":
        custom = input("Path folder output: ").strip().strip("'\"")
        if custom:
            cfg.overwrite = False
            cfg.output_dir = custom
    elif choice == "3":
        if input("Yakin timpa file asli? ketik 'ya': ").strip().lower() == "ya":
            cfg.overwrite = True
            cfg.output_dir = ""


def print_status(cfg: VideoConfig) -> None:
    in_label = cfg.input_path or "(belum diset)"
    out_label = "OVERWRITE" if cfg.overwrite else (cfg.output_dir or "(auto folder)")
    q_label = f"2-pass <= {cfg.target_mb:g} MB" if cfg.target_mb > 0 else f"CRF {cfg.crf}"
    res_label = f"max {cfg.max_height}p" if cfg.max_height > 0 else "asli"
    fps_label = f", <= {cfg.max_fps}fps" if cfg.max_fps > 0 else ""
    print("\n--- VIDEO COMPRESSOR CLI ---")
    print(f"Input   : {in_label}")
    print(f"Output  : {out_label}")
    print(f"Codec   : {cfg.codec.upper()} ({cfg.preset}) | Kualitas: {q_label}")
    print(f"Video   : {res_label}{fps_label} | Audio: {cfg.audio_mode}")
    print("----------------------------")
    print("  1. Set input file / folder")
    print("  2. Set tujuan output")
    print("  3. Codec output (H.264 / H.265 / VP9)")
    print("  4. Kualitas CRF / Target ukuran MB")
    print("  5. Resolusi & batas FPS")
    print("  6. Opsi lanjutan (speed preset, audio, metadata, recursive)")
    print("  7. Mulai kompresi")
    print("  8. Benchmark cepat (tes 10 detik pertama)")
    print("  0. Keluar")


def main() -> None:
    if not FFMPEG_BIN or not FFPROBE_BIN:
        print("ffmpeg / ffprobe belum terinstall. Di Termux jalankan:")
        print("  pkg install ffmpeg")
        sys.exit(1)

    cfg = VideoConfig()
    if len(sys.argv) > 1:
        cfg.input_path = sys.argv[1]

    while True:
        print_status(cfg)
        try:
            choice = input("Menu [0-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar.")
            break

        if choice == "0":
            break
        elif choice == "1":
            raw = input("Path file atau folder video: ").strip().strip("'\"")
            if raw:
                cfg.input_path = raw
        elif choice == "2":
            menu_output(cfg)
        elif choice == "3":
            menu_codec(cfg)
        elif choice == "4":
            menu_quality(cfg)
        elif choice == "5":
            menu_resolution(cfg)
        elif choice == "6":
            menu_advanced(cfg)
        elif choice == "7":
            run_batch(cfg)
        elif choice == "8":
            run_benchmark(cfg)
        else:
            print("[!] Pilihan tidak valid.")


if __name__ == "__main__":
    main()
