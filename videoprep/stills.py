"""Still frames for thumbnails and graphics: at times, at intervals, best-looking, scene cuts."""
from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import fmt_clock, fmt_file, log, write_json
from .media import grab_frame, scene_times


def frame_score(path: Path) -> dict:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    sharp = float(lap.var())
    mean, contrast = float(a.mean()), float(a.std())
    score = np.log(sharp + 1e-6) + 3 * contrast - 4 * abs(mean - 0.45)
    if mean < 0.06 or mean > 0.97 or contrast < 0.04:  # black / blown-out / flat frames
        score -= 10
    return {"score": round(float(score), 3), "sharpness": round(sharp, 5),
            "brightness": round(mean, 3), "contrast": round(contrast, 3)}


def grab_many(src: Path, times: list[float], out_dir: Path, ext: str = "jpg",
              width: int | None = None, workers: int = 4, prefix: str = "") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{prefix}{fmt_file(t)}.{ext}" for t in times]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda tp: grab_frame(src, tp[0], tp[1], width=width), zip(times, paths)))
    return paths


def best_times(src: Path, duration: float, count: int, step: float | None = None,
               skip_edges: float = 3.0) -> list[dict]:
    """Sample candidates across the video, score them, pick the top `count`, spread out."""
    step = step or max(2.0, duration / (count * 10))
    cands = list(np.arange(skip_edges, max(skip_edges + 0.1, duration - skip_edges), step))
    log(f"best: scoring {len(cands)} candidate frames (every {step:.1f}s)")
    with tempfile.TemporaryDirectory() as td:
        paths = grab_many(src, cands, Path(td), width=480)
        scored = [{"time": round(float(t), 3), **frame_score(p)} for t, p in zip(cands, paths)]
    scored.sort(key=lambda d: -d["score"])
    min_gap = duration / (count * 2.5)
    picked: list[dict] = []
    for d in scored:
        if all(abs(d["time"] - p["time"]) >= min_gap for p in picked):
            picked.append(d)
        if len(picked) == count:
            break
    return sorted(picked, key=lambda d: d["time"])


def contact_sheet(items: list[tuple[Path, str]], dst: Path, thumb_w: int = 360, cols: int = 4) -> Path:
    if not items:
        return dst
    thumbs = []
    for p, label in items:
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w, thumb_w * 2))
        thumbs.append((im, label))
    th = max(im.height for im, _ in thumbs)
    pad, lab_h = 8, 26
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (th + lab_h + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = pad + c * (thumb_w + pad), pad + r * (th + lab_h + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + th + 4), label, fill=(230, 230, 230), font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88)
    return dst


def make_stills(src: Path, out_dir: Path, duration: float, *, at: list[float] | None = None,
                every: float | None = None, best: int | None = None, scenes: bool = False,
                threshold: float = 0.3, max_scenes: int = 60, ext: str = "jpg",
                width: int | None = None) -> dict:
    entries: list[dict] = []
    if at:
        entries += [{"time": t, "why": "at"} for t in at]
    if every:
        entries += [{"time": float(t), "why": "every"} for t in np.arange(0, duration, every)]
    if best:
        entries += [{**d, "why": "best"} for d in best_times(src, duration, best)]
    if scenes:
        log("scene scan: decoding the whole video (slow for long/ProRes files)")
        cuts = scene_times(src, threshold, duration=duration)
        cuts.sort(key=lambda c: -c[1])
        # the frame just after a cut is often mid-transition: take it 0.5 s in
        entries += [{"time": min(t + 0.5, duration - 0.1), "why": "scene", "scene_score": round(s, 3)}
                    for t, s in sorted(cuts[:max_scenes])]
    if not entries:
        raise SystemExit("nothing to grab: pass --at, --every, --best and/or --scenes")
    entries.sort(key=lambda e: e["time"])
    log(f"grabbing {len(entries)} full-resolution stills → {out_dir}")
    paths = grab_many(src, [e["time"] for e in entries], out_dir, ext=ext, width=width)
    for e, p in zip(entries, paths):
        e["file"] = p.name
        e["clock"] = fmt_clock(e["time"])
    sheet = contact_sheet([(p, f"{e['clock']}  {e['why']}") for e, p in zip(entries, paths)],
                          out_dir / "_contact.jpg")
    index = {"source": str(src), "stills": entries, "contact_sheet": sheet.name}
    write_json(out_dir / "_index.json", index)
    return index
