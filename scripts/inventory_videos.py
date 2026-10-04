"""Инвентаризация видео (этап 0.4, раздел 6.2): ffprobe -> data/meta/videos.csv.

Структура входной папки: ``<root>/<session>/<...>/<video>.mp4`` — первая подпапка = сессия.
Автоматически заполняются: video_id, session, width, height, fps_nominal, vfr, rotation,
duration, hdr и (по возможности) device. Вручную: color_profile, scene_type, has_people,
device (если не определилось). split заполняется только на этапе 2.3.

Повторный запуск не затирает непустые ручные поля существующих строк.

Пример::

    python scripts/inventory_videos.py D:/videos --out data/meta/videos.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

# Поля из раздела 6.2 + служебные для ручной проверки (rel_path, codec, pix_fmt, color_transfer)
COLUMNS = [
    "video_id", "session", "device", "color_profile", "hdr", "width", "height",
    "fps_nominal", "vfr", "rotation", "duration", "scene_type", "has_people", "split",
    "rel_path", "codec", "pix_fmt", "color_transfer",
]  # fmt: skip
MANUAL_COLUMNS = ["device", "color_profile", "scene_type", "has_people", "split"]
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
# PQ (HDR10/Dolby Vision profile 8.4 и т.п.) и HLG
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
BRAND_RE = re.compile(r"(dji|gopro|insta360|iphone|samsung|xiaomi|pixel)", re.IGNORECASE)


def make_video_id(session: str, stem: str) -> str:
    """Стабильный идентификатор ``<session>__<stem>``: буквы (включая кириллицу), цифры, _ и -."""
    clean = lambda s: re.sub(r"[^\w-]+", "_", s).strip("_")  # noqa: E731
    return f"{clean(session)}__{clean(stem)}"


def find_videos(root: Path) -> list[tuple[str, Path]]:
    """Найти видео в подпапках-сессиях. Файлы прямо в корне пропускаются с предупреждением."""
    found: list[tuple[str, Path]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 2:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {rel} лежит вне папки сессии, пропущен", file=sys.stderr)
            continue
        found.append((rel.parts[0], p))
    return found


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    try:
        val = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return None
    return val if val > 0 else None


def _rotation_cw(stream: dict) -> int:
    """Поворот по часовой стрелке в градусах {0, 90, 180, 270}.

    Display Matrix в ffprobe содержит угол против часовой стрелки (отсюда минус),
    старый тег ``rotate`` — по часовой.
    """
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(-float(sd["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return int(round(float(tag))) % 360
    return 0


def _is_hdr(stream: dict) -> int:
    if stream.get("color_transfer") in HDR_TRANSFERS:
        return 1
    for sd in stream.get("side_data_list", []) or []:
        if "dovi" in str(sd.get("side_data_type", "")).lower():
            return 1
    return 0


def _device_hint(probe: dict) -> str:
    """Best-effort определение устройства по тегам контейнера; пусто, если не уверены."""
    tags = {k.lower(): str(v) for k, v in (probe.get("format", {}).get("tags") or {}).items()}
    for make_key, model_key in [
        ("com.apple.quicktime.make", "com.apple.quicktime.model"),
        ("com.android.manufacturer", "com.android.model"),
        ("make", "model"),
    ]:
        if model_key in tags:
            return " ".join(x for x in (tags.get(make_key, ""), tags[model_key]) if x).strip()
    for v in tags.values():
        m = BRAND_RE.search(v)
        if m:
            return m.group(1).lower()
    return ""


def video_stream(probe: dict) -> dict | None:
    """Первый видеопоток, не являющийся обложкой (attached_pic)."""
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic"):
            return s
    return None


def parse_probe(probe: dict) -> dict[str, object]:
    """Извлечь поля 6.2 из JSON ``ffprobe -show_format -show_streams``."""
    vs = video_stream(probe)
    if vs is None:
        raise ValueError("нет видеопотока")
    fps = _parse_rate(vs.get("avg_frame_rate")) or _parse_rate(vs.get("r_frame_rate"))
    dur = probe.get("format", {}).get("duration") or vs.get("duration")
    return {
        "width": vs.get("width", ""),
        "height": vs.get("height", ""),
        "fps_nominal": round(fps, 3) if fps else "",
        "rotation": _rotation_cw(vs),
        "duration": round(float(dur), 3) if dur not in (None, "N/A") else "",
        "hdr": _is_hdr(vs),
        "device": _device_hint(probe),
        "codec": vs.get("codec_name", ""),
        "pix_fmt": vs.get("pix_fmt", ""),
        "color_transfer": vs.get("color_transfer", ""),
    }


def detect_vfr(pts: Sequence[float], tol: float = 0.1, max_irregular: float = 0.01) -> int:
    """VFR, если доля интервалов между кадрами, отклоняющихся от медианы более чем на ``tol``
    (относительно), превышает ``max_irregular``. Пороги — эвристика."""
    t = np.sort(np.asarray(pts, dtype=float))
    if len(t) < 3:
        return 0
    d = np.diff(t)
    d = d[d > 0]
    if len(d) == 0:
        return 0
    med = np.median(d)
    irregular = np.mean(np.abs(d - med) > tol * med)
    return int(irregular > max_irregular)


def run_ffprobe(path: Path) -> dict:
    """JSON с форматом и потоками."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def read_video_pts(path: Path) -> list[float]:
    """PTS (секунды) всех пакетов первого видеопотока; только демультиплексирование."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )  # fmt: skip
    pts = []
    for line in out.stdout.splitlines():
        line = line.strip().rstrip(",")
        try:
            pts.append(float(line))
        except ValueError:
            continue  # N/A
    return pts


def merge_existing(new_rows: list[dict], old_rows: list[dict]) -> list[dict]:
    """Сохранить непустые ручные поля из старого CSV; строки пропавших видео оставить."""
    old = {r["video_id"]: r for r in old_rows}
    merged = []
    for r in new_rows:
        prev = old.pop(r["video_id"], None)
        if prev:
            for c in MANUAL_COLUMNS:
                if str(prev.get(c, "")).strip():
                    r[c] = prev[c]
        merged.append(r)
    for vid, r in old.items():
        print(f"ПРЕДУПРЕЖДЕНИЕ: {vid} есть в CSV, но файл не найден; строка сохранена",
              file=sys.stderr)  # fmt: skip
        merged.append({c: r.get(c, "") for c in COLUMNS})
    return sorted(merged, key=lambda r: (r["session"], r["video_id"]))


def summarize(rows: list[dict]) -> str:
    """Краткая сводка против целей K0 (>= 60 видео, >= 15 сессий, >= 2 устройства)."""
    sessions = {r["session"] for r in rows}
    devices = {r["device"] for r in rows if r["device"]}
    durs = [float(r["duration"]) for r in rows if r["duration"] != ""]
    long = sum(d > 300 for d in durs)
    n_hdr = sum(int(r["hdr"] or 0) for r in rows)
    n_vfr = sum(int(r["vfr"] or 0) for r in rows)
    lines = [
        f"Видео: {len(rows)} (цель >= 60)",
        f"Сессий: {len(sessions)} (цель >= 15)",
        f"Устройств (по тегам): {len(devices)} {sorted(devices)} (цель >= 2; пустые — заполнить)",
        f"Суммарно: {sum(durs) / 60:.1f} мин; длиннее 5 мин: {long}",
        f"HDR: {n_hdr}; VFR: {n_vfr}",
        f"Без device: {sum(not r['device'] for r in rows)}; "
        "color_profile/scene_type/has_people — заполнить вручную",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Инвентаризация видео через ffprobe")
    ap.add_argument("root", type=Path, help="папка с видео, подпапки = сессии")
    ap.add_argument("--out", type=Path, default=Path("data/meta/videos.csv"))
    ap.add_argument("--no-vfr-check", action="store_true", help="не читать PTS пакетов (быстрее)")
    args = ap.parse_args(argv)

    if shutil.which("ffprobe") is None:
        print("ffprobe не найден в PATH. Windows: winget install Gyan.FFmpeg", file=sys.stderr)
        return 2
    videos = find_videos(args.root)
    if not videos:
        print(f"В {args.root} не найдено видео в подпапках-сессиях", file=sys.stderr)
        return 1

    rows, seen = [], {}
    for session, path in videos:
        vid = make_video_id(session, path.stem)
        rel = path.relative_to(args.root).as_posix()
        if vid in seen:
            print(f"ОШИБКА: одинаковый video_id {vid}: {seen[vid]} и {rel}", file=sys.stderr)
            return 1
        seen[vid] = rel
        row = dict.fromkeys(COLUMNS, "")
        row.update(video_id=vid, session=session, rel_path=rel)
        try:
            row.update(parse_probe(run_ffprobe(path)))
            row["vfr"] = "" if args.no_vfr_check else detect_vfr(read_video_pts(path))
        except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as e:
            print(f"ОШИБКА ffprobe для {rel}: {e}", file=sys.stderr)
        rows.append(row)
        print(
            f"{rel}: {row['width']}x{row['height']} {row['fps_nominal']} fps "
            f"rot={row['rotation']} hdr={row['hdr']} vfr={row['vfr']} {row['device']!r}"
        )

    if args.out.exists():
        with args.out.open(encoding="utf-8", newline="") as f:
            rows = merge_existing(rows, list(csv.DictReader(f)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {args.out}\n{summarize(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
