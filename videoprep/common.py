"""Shared helpers: paths, jobs, timestamps, logging, subprocess."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(os.environ.get("VIDEOPREP_WORK", "work"))
MODELS = Path(os.environ.get("VIDEOPREP_MODELS", "models"))

_T0 = time.time()


def log(msg: str) -> None:
    el = time.time() - _T0
    print(f"[{int(el // 60):3d}m{int(el % 60):02d}s] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=capture, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2000:]
        die(f"command failed ({p.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return p


# ---------------------------------------------------------------- timestamps

def fmt_srt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt(t: float) -> str:
    return fmt_srt(t).replace(",", ".")


def fmt_clock(t: float) -> str:
    t = int(max(0.0, t))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_file(t: float) -> str:
    """Filesystem-safe timestamp, e.g. 00h12m03s500."""
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"


def parse_ts(s: str) -> float:
    """Accept 83, 83.5, 83s, 1:23, 01:02:03.5, 01:02:03,500, 2m, 1h2m3s."""
    s = s.strip().lower().replace(",", ".")
    if re.fullmatch(r"[\d.]+s?", s):
        return float(s.rstrip("s"))
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", s)
    if m and any(m.groups()):
        h, mi, se = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(se or 0)
    parts = s.split(":")
    if 2 <= len(parts) <= 3:
        parts = [float(p) for p in parts]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad timestamp: {s!r}")


# ---------------------------------------------------------------- jobs

def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    s = re.sub(r"-+", "-", s)
    return s[:80] or "job"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, name: str):
        self.name = name
        self.dir = WORK / name

    # paths -------------------------------------------------------------
    @property
    def meta_path(self) -> Path: return self.dir / "job.json"
    @property
    def audio(self) -> Path: return self.dir / "audio.wav"
    @property
    def asr_path(self) -> Path: return self.dir / "asr.json"
    @property
    def asr_partial(self) -> Path: return self.dir / "asr.partial.jsonl"
    @property
    def diar_path(self) -> Path: return self.dir / "diarization.json"
    @property
    def speakers_path(self) -> Path: return self.dir / "speakers.json"
    @property
    def out(self) -> Path: return self.dir / "out"
    @property
    def stills(self) -> Path: return self.dir / "stills"

    def exists(self) -> bool:
        return self.meta_path.exists()

    def load(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def save(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta["updated"] = now_iso()
        write_json(self.meta_path, meta)

    def update(self, **kw) -> dict:
        meta = self.load()
        meta.update(kw)
        self.save(meta)
        return meta

    @property
    def source(self) -> Path:
        return Path(self.load()["source"])

    @property
    def base(self) -> str:
        """Output file stem: the source file's stem."""
        return Path(self.load()["source"]).stem


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def resolve_job(arg: str, create: bool = False, job_name: str | None = None) -> Job:
    """`arg` is a job name, a work/<job> path, or a video file path.

    A video path maps to a job named after the file (or `job_name`); with
    create=True the job is created if missing.
    """
    p = Path(arg)
    if not p.is_file():
        name = p.name if p.parent.name == WORK.name else arg
        j = Job(name)
        if j.exists():
            return j
        die(f"no such job or file: {arg} (jobs live in {WORK}/; run `vp jobs`)")
    j = Job(job_name or slugify(p.stem))
    if j.exists():
        src = j.load().get("source")
        if src and Path(src).resolve() != p.resolve():
            die(f"job {j.name} already exists for a different source ({src}); pass --job NAME")
        return j
    if not create:
        die(f"no job for {arg} yet; run `vp transcribe {arg}` first")
    j.save({"name": j.name, "source": str(p), "created": now_iso()})
    return j
