"""ffmpeg/ffprobe wrappers: probe, audio extraction, frame grabs, scene scan."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .common import log, run


def probe(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)])
    return json.loads(p.stdout)


def _fps(s: str | None) -> float | None:
    if not s or s in ("0/0",):
        return None
    if "/" in s:
        a, b = s.split("/")
        return round(float(a) / float(b), 3) if float(b) else None
    return float(s)


def summarize(info: dict) -> dict:
    fmt = info.get("format", {})
    out = {
        "duration": float(fmt.get("duration", 0) or 0),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "container": fmt.get("format_name"),
        "video": [],
        "audio": [],
    }
    a_idx = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic"):
            out["video"].append({
                "codec": s.get("codec_name"), "profile": s.get("profile"),
                "width": s.get("width"), "height": s.get("height"),
                "fps": _fps(s.get("avg_frame_rate") or s.get("r_frame_rate")),
                "pix_fmt": s.get("pix_fmt"), "color_transfer": s.get("color_transfer"),
            })
        elif s.get("codec_type") == "audio":
            out["audio"].append({
                "audio_index": a_idx, "codec": s.get("codec_name"),
                "channels": s.get("channels"), "layout": s.get("channel_layout"),
                "sample_rate": int(s.get("sample_rate", 0) or 0),
                "language": s.get("tags", {}).get("language"),
                "title": s.get("tags", {}).get("title"),
            })
            a_idx += 1
    return out


def extract_audio(src: Path, dst: Path, stream: int = 0, channel: str | None = None) -> None:
    """16 kHz mono PCM for ASR/diarization.

    channel: None = downmix all channels; 'L'/'R' or an integer index picks one
    (useful when two lav mics are recorded to separate channels).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = []
    if channel is not None:
        c = {"l": 0, "left": 0, "r": 1, "right": 1}.get(str(channel).lower(), channel)
        af = ["-af", f"pan=mono|c0=c{int(c)}"]
    tmp = dst.with_suffix(".tmp.wav")
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-map", f"0:a:{stream}", "-vn", *af,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(dst)


def grab_frame(src: Path, t: float, dst: Path, width: int | None = None,
               quality: int = 2) -> Path:
    """One frame at time t (fast input seek). width=None keeps full resolution."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"scale={width}:-2:flags=lanczos"] if width else []
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1", "-an"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", str(quality)]
    cmd.append(str(dst))
    run(cmd)
    return dst


def scene_times(src: Path, threshold: float = 0.3, scan_width: int = 320,
                duration: float | None = None) -> list[tuple[float, float]]:
    """Decode the whole video and return [(time, score)] for scene cuts.

    Slow on long/high-bitrate files (full decode); logs progress.
    """
    vf = f"scale={scan_width}:-2,select='gt(scene\\,{threshold})',metadata=print"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(src), "-an",
           "-vf", vf, "-f", "null", "-"]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
    cuts: list[tuple[float, float]] = []
    pending_t = None
    last_report = 0.0
    for line in proc.stderr:  # type: ignore[union-attr]
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            pending_t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([\d.]+)", line)
        if m and pending_t is not None:
            cuts.append((pending_t, float(m.group(1))))
            pending_t = None
            continue
        m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
        if m and duration:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if t - last_report > max(60.0, duration / 20):
                last_report = t
                log(f"scene scan {100 * t / duration:4.1f}% ({len(cuts)} cuts so far)")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg scene scan failed")
    return cuts
