#!/usr/bin/env python3
"""Interactive CLI photo compressor with sharp downscaling and format conversion."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import os
import shutil
import sys
import time

try:
    from PIL import Image, ImageFile, ImageFilter, ImageOps
except ImportError:
    print("Pillow belum terinstall. Di Termux jalankan:")
    print("  pkg install python-pillow")
    print("atau: pip install pillow")
    sys.exit(1)

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Populate Image.SAVE before probing codec availability (preinit skips WebP/AVIF plugins).
Image.init()

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
WEBP_SUPPORTED = "WEBP" in Image.SAVE
AVIF_SUPPORTED = "AVIF" in Image.SAVE

# Pillow 13+ ships Magic Kernel Sharp (MKS2021), which adds built-in sharpening on downscale.
RESAMPLE_FILTERS: dict[str, tuple[int, str]] = {
    "lanczos": (Image.Resampling.LANCZOS, "Lanczos (tajam, standar foto)"),
    "hamming": (Image.Resampling.HAMMING, "Hamming (cepat, tanpa ringing)"),
    "bicubic": (Image.Resampling.BICUBIC, "Bicubic (halus, cocok foto noisy)"),
    "box": (Image.Resampling.BOX, "Box (area average, thumbnail cepat)"),
}
if hasattr(Image.Resampling, "MKS2021"):
    RESAMPLE_FILTERS = {
        "mks2021": (Image.Resampling.MKS2021, "Magic Kernel Sharp 2021 (tajam, minim ringing)"),
        **RESAMPLE_FILTERS,
    }


@dataclass
class CompressConfig:
    input_path: str = ""
    output_dir: str = ""
    overwrite: bool = False
    output_format: str = "keep"  # keep | jpeg | webp | png | avif
    quality: int = 85
    target_kb: int = 0  # 0 = off, >0 = binary search quality to fit target KB
    max_side: int = 0  # 0 = keep original resolution
    resample: str = "mks2021" if "mks2021" in RESAMPLE_FILTERS else "lanczos"
    sharpen_after_resize: bool = True
    jpeg_subsampling: str = "4:2:0"  # 4:4:4 (tajam teks/merah) | 4:2:0 (lebih kecil)
    webp_lossless: bool = False
    png_quantize: bool = False
    strip_exif: bool = True
    keep_icc: bool = True
    recursive: bool = False
    skip_if_larger: bool = True
    workers: int = min(4, os.cpu_count() or 2)


@dataclass
class ImageOutcome:
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


def resolve_format(src: Path, requested: str) -> str:
    if requested != "keep":
        return requested.upper()
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".webp" and WEBP_SUPPORTED:
        return "WEBP"
    if ext == ".png":
        return "PNG"
    if ext == ".avif" and AVIF_SUPPORTED:
        return "AVIF"
    return "JPEG"


def format_extension(fmt: str) -> str:
    return {
        "JPEG": ".jpg",
        "WEBP": ".webp",
        "PNG": ".png",
        "AVIF": ".avif",
    }[fmt]


def has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)


def flatten_for_jpeg(im: Image.Image) -> Image.Image:
    if not has_alpha(im):
        return im.convert("RGB") if im.mode != "RGB" else im
    rgba = im.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def downscale_image(im: Image.Image, cfg: CompressConfig) -> tuple[Image.Image, bool]:
    if cfg.max_side <= 0:
        return im, False
    w, h = im.size
    longest = max(w, h)
    if longest <= cfg.max_side:
        return im, False

    scale = cfg.max_side / longest
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    filt = RESAMPLE_FILTERS[cfg.resample][0]

    # Palette images force NEAREST in Pillow resize; promote before resampling.
    if im.mode == "P":
        im = im.convert("RGBA" if has_alpha(im) else "RGB")

    # reducing_gap=3.0 matches full-pass resampling visually while running faster on large photos.
    resized = im.resize(new_size, resample=filt, reducing_gap=3.0)

    # MKS2021 already embeds a sharpening kernel; avoid double-sharpening halos.
    if cfg.sharpen_after_resize and cfg.resample != "mks2021":
        resized = resized.filter(ImageFilter.UnsharpMask(radius=0.8, percent=55, threshold=3))

    return resized, True


def encode_image(
    im: Image.Image,
    fmt: str,
    quality: int,
    cfg: CompressConfig,
    icc_profile: bytes | None,
    exif_bytes: bytes | None,
) -> bytes:
    buf = BytesIO()
    meta_kwargs: dict = {}
    if cfg.keep_icc and icc_profile:
        meta_kwargs["icc_profile"] = icc_profile
    if not cfg.strip_exif and exif_bytes and fmt in ("JPEG", "WEBP", "AVIF"):
        meta_kwargs["exif"] = exif_bytes

    if fmt == "JPEG":
        out = flatten_for_jpeg(im)
        # Pillow docs warn JPEG quality > 95 inflates size with negligible visual gain.
        q = max(1, min(95, quality))
        sub = 0 if cfg.jpeg_subsampling == "4:4:4" else 2
        out.save(
            buf,
            format="JPEG",
            quality=q,
            optimize=True,
            progressive=True,
            subsampling=sub,
            **meta_kwargs,
        )
    elif fmt == "WEBP":
        out = im.convert("RGBA") if has_alpha(im) else im.convert("RGB")
        if cfg.webp_lossless:
            out.save(buf, format="WEBP", lossless=True, quality=100, method=6, **meta_kwargs)
        else:
            out.save(
                buf,
                format="WEBP",
                quality=max(1, min(100, quality)),
                method=6,
                **meta_kwargs,
            )
    elif fmt == "AVIF":
        out = im.convert("RGBA") if has_alpha(im) else im.convert("RGB")
        sub = "4:4:4" if cfg.jpeg_subsampling == "4:4:4" else "4:2:0"
        out.save(
            buf,
            format="AVIF",
            quality=max(1, min(100, quality)),
            subsampling=sub,
            speed=6,
            **meta_kwargs,
        )
    elif fmt == "PNG":
        out = im
        if cfg.png_quantize:
            base = im.convert("RGBA") if has_alpha(im) else im.convert("RGB")
            out = base.quantize(
                colors=256,
                method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
        out.save(buf, format="PNG", optimize=True, compress_level=9, **meta_kwargs)
    else:
        raise ValueError(f"Format tidak didukung: {fmt}")

    return buf.getvalue()


def encode_with_target_kb(
    im: Image.Image,
    fmt: str,
    cfg: CompressConfig,
    icc_profile: bytes | None,
    exif_bytes: bytes | None,
) -> tuple[bytes, int]:
    limit = cfg.target_kb * 1024
    lo, hi = 30, (95 if fmt == "JPEG" else 96)
    best_payload = encode_image(im, fmt, hi, cfg, icc_profile, exif_bytes)
    if len(best_payload) <= limit or fmt == "PNG" or (fmt == "WEBP" and cfg.webp_lossless):
        return best_payload, hi

    best_q = lo
    best_fit: bytes | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = encode_image(im, fmt, mid, cfg, icc_profile, exif_bytes)
        if len(candidate) <= limit:
            best_fit = candidate
            best_q = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_fit is not None:
        return best_fit, best_q
    return encode_image(im, fmt, 30, cfg, icc_profile, exif_bytes), 30


def compress_one(src: Path, dst_dir: Path | None, cfg: CompressConfig) -> ImageOutcome:
    orig_bytes = src.stat().st_size
    try:
        with Image.open(src) as opened:
            icc_profile = opened.info.get("icc_profile")
            exif_bytes = opened.info.get("exif")
            # Apply EXIF rotation before stripping metadata so portrait photos stay upright.
            im = ImageOps.exif_transpose(opened)
            im.load()

        fmt = resolve_format(src, cfg.output_format)
        im, was_resized = downscale_image(im, cfg)

        if cfg.target_kb > 0:
            payload, used_q = encode_with_target_kb(im, fmt, cfg, icc_profile, exif_bytes)
            note = f"{fmt} q={used_q}"
        else:
            payload = encode_image(im, fmt, cfg.quality, cfg, icc_profile, exif_bytes)
            note = f"{fmt} q={cfg.quality}" if fmt != "PNG" else fmt

        if was_resized:
            note += f" {im.width}x{im.height}"

        same_fmt = src.suffix.lower() == format_extension(fmt) or (
            fmt == "JPEG" and src.suffix.lower() in (".jpg", ".jpeg")
        )
        out_name = src.name if same_fmt else f"{src.stem}{format_extension(fmt)}"
        dst = (src.parent / out_name) if cfg.overwrite else (dst_dir / out_name)

        if cfg.skip_if_larger and same_fmt and not was_resized and len(payload) >= orig_bytes:
            if not cfg.overwrite:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return ImageOutcome(src, dst, orig_bytes, orig_bytes, "skipped", "asli lebih kecil")

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
        tmp_dst.write_bytes(payload)
        tmp_dst.replace(dst)

        if cfg.overwrite and dst != src and src.exists():
            src.unlink()

        return ImageOutcome(src, dst, orig_bytes, len(payload), "ok", note)
    except Exception as exc:
        return ImageOutcome(src, None, orig_bytes, orig_bytes, "error", str(exc))


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTS else []
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "compressed"
    return input_path.parent / f"{input_path.name}_compressed"


def run_batch(cfg: CompressConfig) -> None:
    if not cfg.input_path:
        print("\n[!] Input path belum diisi. Pilih menu 1 dulu.")
        return

    src_root = Path(cfg.input_path).expanduser().resolve()
    if not src_root.exists():
        print(f"\n[!] Path tidak ditemukan: {src_root}")
        return

    files = collect_images(src_root, cfg.recursive)
    if not files:
        print(f"\n[!] Tidak ada file foto yang didukung di: {src_root}")
        return

    out_root = None if cfg.overwrite else (
        Path(cfg.output_dir).expanduser().resolve() if cfg.output_dir else default_output_dir(src_root)
    )

    print(f"\nMemproses {len(files)} foto ({cfg.workers} worker)...")
    if out_root:
        print(f"Output: {out_root}")
    else:
        print("Output: OVERWRITE file asli")

    t0 = time.monotonic()
    total_orig = 0
    total_new = 0
    ok_count = 0
    skip_count = 0
    err_count = 0

    def job(img_path: Path) -> ImageOutcome:
        if out_root is None:
            return compress_one(img_path, None, cfg)
        rel_parent = img_path.parent.relative_to(src_root) if src_root.is_dir() else Path(".")
        return compress_one(img_path, out_root / rel_parent, cfg)

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(job, f): f for f in files}
        for idx, fut in enumerate(as_completed(futures), 1):
            out = fut.result()
            total_orig += out.orig_bytes
            total_new += out.new_bytes
            name = out.src.name
            if out.status == "ok":
                ok_count += 1
                pct = (1 - out.new_bytes / out.orig_bytes) * 100 if out.orig_bytes else 0.0
                print(
                    f"  [{idx}/{len(files)}] OK   {name}: "
                    f"{fmt_size(out.orig_bytes)} -> {fmt_size(out.new_bytes)} "
                    f"({pct:+.1f}%) [{out.detail}]"
                )
            elif out.status == "skipped":
                skip_count += 1
                print(f"  [{idx}/{len(files)}] SKIP {name}: {out.detail}")
            else:
                err_count += 1
                print(f"  [{idx}/{len(files)}] ERR  {name}: {out.detail}")

    elapsed = time.monotonic() - t0
    saved = total_orig - total_new
    saved_pct = (saved / total_orig * 100) if total_orig else 0.0
    print(
        f"\nSelesai dalam {elapsed:.2f}s | OK: {ok_count}, Skip: {skip_count}, Error: {err_count}\n"
        f"Total: {fmt_size(total_orig)} -> {fmt_size(total_new)} "
        f"(hemat {fmt_size(saved)} / {saved_pct:.1f}%)"
    )


def run_benchmark(cfg: CompressConfig) -> None:
    sample_raw = input("Path 1 foto sample (kosong = pakai input saat ini): ").strip()
    path = Path(sample_raw or cfg.input_path).expanduser().resolve()
    if path.is_dir():
        imgs = collect_images(path, False)
        if not imgs:
            print("[!] Tidak ada foto di folder tersebut.")
            return
        path = imgs[0]
    if not path.is_file():
        print(f"[!] File tidak ditemukan: {path}")
        return

    orig_bytes = path.stat().st_size
    with Image.open(path) as opened:
        icc = opened.info.get("icc_profile")
        base = ImageOps.exif_transpose(opened)
        base.load()

    resized, _ = downscale_image(base, cfg)
    print(f"\nBenchmark: {path.name} ({base.width}x{base.height}, asli {fmt_size(orig_bytes)})")
    if cfg.max_side > 0:
        print(f"Resize aktif: {resized.width}x{resized.height} ({cfg.resample})")

    cases = [
        ("JPEG q=85 4:2:0", "JPEG", 85, "4:2:0", False),
        ("JPEG q=85 4:4:4", "JPEG", 85, "4:4:4", False),
    ]
    if WEBP_SUPPORTED:
        cases.extend([
            ("WEBP q=85", "WEBP", 85, "4:2:0", False),
            ("WEBP q=82", "WEBP", 82, "4:2:0", False),
        ])
    cases.extend([
        ("PNG opt", "PNG", 85, "4:2:0", False),
        ("PNG quant 256", "PNG", 85, "4:2:0", True),
    ])
    if AVIF_SUPPORTED:
        cases.append(("AVIF q=65", "AVIF", 65, "4:2:0", False))

    for label, fmt, q, sub, quant in cases:
        t0 = time.monotonic()
        tmp_cfg = CompressConfig(
            jpeg_subsampling=sub,
            png_quantize=quant,
            keep_icc=cfg.keep_icc,
            strip_exif=True,
        )
        try:
            payload = encode_image(resized, fmt, q, tmp_cfg, icc, None)
        except (KeyError, OSError, ValueError) as exc:
            print(f"  {label:<16} -> GAGAL ({exc})")
            continue
        ms = (time.monotonic() - t0) * 1000
        pct = (1 - len(payload) / orig_bytes) * 100
        print(f"  {label:<16} -> {fmt_size(len(payload)):>9} ({pct:+5.1f}%)  {ms:6.0f} ms")

    missing = [name for name, ok in (("WEBP", WEBP_SUPPORTED), ("AVIF", AVIF_SUPPORTED)) if not ok]
    if missing:
        print(f"  [info] Codec tidak tersedia di build Pillow ini: {', '.join(missing)}")


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


def menu_format(cfg: CompressConfig) -> None:
    print("\nFormat Output:")
    print("  1. Keep asli (JPG->JPG, PNG->PNG, WebP->WebP)")
    if WEBP_SUPPORTED:
        print("  2. WebP (rekomendasi: kecil & tajam, support transparan)")
    else:
        print("  2. WebP (tidak tersedia di build Pillow ini)")
    print("  3. JPEG (universal, progressive + optimize)")
    print("  4. PNG (lossless / bisa quantize 256 warna)")
    if AVIF_SUPPORTED:
        print("  5. AVIF (paling kecil, butuh encoder lebih lambat)")
    else:
        print("  5. AVIF (tidak tersedia di build Pillow ini)")

    choice = input("Pilih [1-5]: ").strip()
    mapping = {"1": "keep", "2": "webp", "3": "jpeg", "4": "png", "5": "avif"}
    picked = mapping.get(choice)
    if picked == "webp" and not WEBP_SUPPORTED:
        print("[!] Pillow di environment ini belum mendukung WebP.")
        return
    if picked == "avif" and not AVIF_SUPPORTED:
        print("[!] Pillow di environment ini belum mendukung AVIF.")
        return
    if picked:
        cfg.output_format = picked


def menu_quality(cfg: CompressConfig) -> None:
    print("\nPreset Kualitas:")
    print("  1. Ultra Jernih (quality 90)")
    print("  2. Balanced Jernih (quality 85 - rekomendasi)")
    print("  3. Hemat Tajam (quality 78)")
    print("  4. Custom quality (1-95)")
    print("  5. Target ukuran maksimal (KB) via binary search")
    choice = input("Pilih [1-5]: ").strip()
    if choice == "1":
        cfg.quality, cfg.target_kb = 90, 0
    elif choice == "2":
        cfg.quality, cfg.target_kb = 85, 0
    elif choice == "3":
        cfg.quality, cfg.target_kb = 78, 0
    elif choice == "4":
        cfg.quality = prompt_int("Quality (1-95)", cfg.quality, 1, 95)
        cfg.target_kb = 0
    elif choice == "5":
        cfg.target_kb = prompt_int("Target max KB (misal 500)", max(50, cfg.target_kb or 500), 10, 50000)


def menu_resize(cfg: CompressConfig) -> None:
    print("\nSmart Resize (sisi terpanjang, proporsi tetap):")
    print("  1. Mati (pertahankan resolusi asli)")
    print("  2. Max 2560 px (2.5K - tajam buat layar HP/monitor)")
    print("  3. Max 2048 px (standar share/web jernih)")
    print("  4. Max 1920 px (Full HD)")
    print("  5. Max 1280 px (ringan)")
    print("  6. Custom pixel")
    print("  7. Ganti algoritma filter resize")
    print(f"  8. Toggle UnsharpMask tipis setelah resize (saat ini: {'ON' if cfg.sharpen_after_resize else 'OFF'})")

    choice = input("Pilih [1-8]: ").strip()
    sizes = {"1": 0, "2": 2560, "3": 2048, "4": 1920, "5": 1280}
    if choice in sizes:
        cfg.max_side = sizes[choice]
    elif choice == "6":
        cfg.max_side = prompt_int("Max sisi terpanjang (px)", cfg.max_side or 2048, 128, 16000)
    elif choice == "7":
        keys = list(RESAMPLE_FILTERS.keys())
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {key:<8} - {RESAMPLE_FILTERS[key][1]}")
        pick = prompt_int(f"Pilih filter [1-{len(keys)}]", 1, 1, len(keys))
        cfg.resample = keys[pick - 1]
    elif choice == "8":
        cfg.sharpen_after_resize = not cfg.sharpen_after_resize


def menu_advanced(cfg: CompressConfig) -> None:
    while True:
        print("\nOpsi Lanjutan:")
        print(f"  1. JPEG/AVIF subsampling : {cfg.jpeg_subsampling} (4:4:4 = tajam teks, 4:2:0 = lebih kecil)")
        print(f"  2. WebP lossless         : {'ON' if cfg.webp_lossless else 'OFF'}")
        print(f"  3. PNG quantize 256      : {'ON' if cfg.png_quantize else 'OFF'}")
        print(f"  4. Strip EXIF metadata   : {'ON' if cfg.strip_exif else 'OFF'}")
        print(f"  5. Pertahankan ICC color : {'ON' if cfg.keep_icc else 'OFF'}")
        print(f"  6. Scan subfolder        : {'ON' if cfg.recursive else 'OFF'}")
        print(f"  7. Skip jika >= asli     : {'ON' if cfg.skip_if_larger else 'OFF'}")
        print(f"  8. Jumlah worker thread  : {cfg.workers}")
        print("  0. Kembali")
        choice = input("Pilih [0-8]: ").strip()
        if choice == "0" or not choice:
            return
        elif choice == "1":
            cfg.jpeg_subsampling = "4:4:4" if cfg.jpeg_subsampling == "4:2:0" else "4:2:0"
        elif choice == "2":
            cfg.webp_lossless = not cfg.webp_lossless
        elif choice == "3":
            cfg.png_quantize = not cfg.png_quantize
        elif choice == "4":
            cfg.strip_exif = not cfg.strip_exif
        elif choice == "5":
            cfg.keep_icc = not cfg.keep_icc
        elif choice == "6":
            cfg.recursive = not cfg.recursive
        elif choice == "7":
            cfg.skip_if_larger = not cfg.skip_if_larger
        elif choice == "8":
            cfg.workers = prompt_int("Worker threads", cfg.workers, 1, 16)


def menu_output(cfg: CompressConfig) -> None:
    print("\nTujuan Output:")
    print("  1. Folder otomatis (<folder>_compressed / compressed)")
    print("  2. Folder custom")
    print("  3. Overwrite file asli (hati-hati, tidak bisa undo)")
    choice = input("Pilih [1-3]: ").strip()
    if choice == "1":
        cfg.overwrite = False
        cfg.output_dir = ""
    elif choice == "2":
        custom = input("Path folder output: ").strip()
        if custom:
            cfg.overwrite = False
            cfg.output_dir = custom
    elif choice == "3":
        confirm = input("Yakin timpa file asli? ketik 'ya': ").strip().lower()
        if confirm == "ya":
            cfg.overwrite = True
            cfg.output_dir = ""


def print_status(cfg: CompressConfig) -> None:
    in_label = cfg.input_path or "(belum diset)"
    out_label = "OVERWRITE" if cfg.overwrite else (cfg.output_dir or "(auto folder)")
    q_label = f"target <= {cfg.target_kb} KB" if cfg.target_kb > 0 else f"q={cfg.quality}"
    r_label = f"max {cfg.max_side}px ({cfg.resample})" if cfg.max_side > 0 else f"OFF ({cfg.resample})"
    print("\n--- PHOTO COMPRESSOR CLI ---")
    print(f"Input   : {in_label}")
    print(f"Output  : {out_label}")
    print(f"Format  : {cfg.output_format.upper()} | Kualitas: {q_label}")
    print(f"Resize  : {r_label} | Sharpen: {'ON' if cfg.sharpen_after_resize else 'OFF'}")
    print(f"Meta    : EXIF={'strip' if cfg.strip_exif else 'keep'}, ICC={'keep' if cfg.keep_icc else 'strip'}")
    print("----------------------------")
    print("  1. Set input file / folder")
    print("  2. Set tujuan output")
    print("  3. Format output (asli / WebP / JPEG / PNG / AVIF)")
    print("  4. Kualitas / Target ukuran KB")
    print("  5. Resize & filter downscale")
    print("  6. Opsi lanjutan (subsampling, EXIF, recursive, PNG quant)")
    print("  7. Mulai kompresi")
    print("  8. Benchmark cepat (bandingkan format pada 1 foto)")
    print("  0. Keluar")


def main() -> None:
    cfg = CompressConfig()
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
            raw = input("Path file atau folder foto: ").strip().strip("'\"")
            if raw:
                cfg.input_path = raw
        elif choice == "2":
            menu_output(cfg)
        elif choice == "3":
            menu_format(cfg)
        elif choice == "4":
            menu_quality(cfg)
        elif choice == "5":
            menu_resize(cfg)
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
