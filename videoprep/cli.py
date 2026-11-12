"""`vp` command line. Run via the ./vp wrapper (Docker) — see README.md / CLAUDE.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import media
from .common import (WORK, Job, die, fmt_clock, log, now_iso, parse_ts, read_json,
                     resolve_job, write_json)
from .transcribe import DEFAULT_MODEL, model_id


# ---------------------------------------------------------------- helpers

def _audio_step(job: Job, args) -> dict:
    meta = job.load()
    if "media" not in meta:
        meta["media"] = media.summarize(media.probe(job.source))
        job.save(meta)
    m = meta["media"]
    if not m["audio"]:
        die("source has no audio stream")
    if len(m["audio"]) > 1 or (m["audio"][0]["channels"] or 1) > 2:
        log(f"note: {len(m['audio'])} audio stream(s); channels per stream "
            f"{[a['channels'] for a in m['audio']]}. Using stream {args.audio_stream}"
            + (f", channel {args.channel}" if args.channel is not None else ", downmixed"))
    want = {"stream": args.audio_stream, "channel": args.channel}
    if not job.audio.exists() or args.force or meta.get("audio_opts") != want:
        log(f"extracting audio → {job.audio}")
        media.extract_audio(job.source, job.audio, args.audio_stream, args.channel)
        meta = job.update(audio_opts=want)
    return meta


def _diar_kwargs(args) -> dict:
    return {"num_speakers": args.speakers, "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers}


def _run_diarize(job: Job, args) -> None:
    from .diarize import diarize

    if args.diarizer == "none":
        job.diar_path.unlink(missing_ok=True)
        job.speakers_path.unlink(missing_ok=True)
        return
    asr = read_json(job.asr_path)
    res = diarize(job.audio, asr, method=args.diarizer, cache=job.dir / "embeddings.npz",
                  **_diar_kwargs(args))
    res["created"] = now_iso()
    res["requested"] = _diar_kwargs(args)
    write_json(job.diar_path, res)
    build_speaker_review(job, asr, res)


def _names(job: Job) -> dict[str, str]:
    if not job.speakers_path.exists():
        return {}
    sp = read_json(job.speakers_path)
    return {k: v["name"] for k, v in sp["speakers"].items() if v.get("name")}


def build_speaker_review(job: Job, asr: dict, diar: dict) -> dict:
    """speakers.json + one still per speaker (taken while they talk) + quotes."""
    from .render import build_words, paragraphs
    from .stills import contact_sheet

    words = build_words(asr, diar, None)
    paras = paragraphs(words)
    old = read_json(job.speakers_path) if job.speakers_path.exists() else {}
    keep_names = old.get("diarization_created") == diar.get("created")
    speakers: dict[str, dict] = {}
    total = sum(w["end"] - w["start"] for w in words) or 1
    for spk in sorted({w["speaker_id"] for w in words}):
        ws = [w for w in words if w["speaker_id"] == spk]
        ps = sorted([p for p in paras if p["speaker"] == spk], key=lambda p: -(p["end"] - p["start"]))
        talk = sum(w["end"] - w["start"] for w in ws)
        longest = ps[0] if ps else {"start": ws[0]["start"], "end": ws[0]["end"]}
        t = (longest["start"] + longest["end"]) / 2
        still = job.dir / "speakers" / f"{spk}.jpg"
        if job.load().get("media", {}).get("video"):
            media.grab_frame(job.source, t, still, width=960)
        speakers[spk] = {
            "name": (old.get("speakers", {}).get(spk, {}).get("name") if keep_names else None),
            "talk_time_sec": round(talk, 1), "share": round(talk / total, 3),
            "words": len(ws), "first_heard": fmt_clock(ws[0]["start"]),
            "still": str(still.relative_to(job.dir)) if still.exists() else None,
            "still_time": fmt_clock(t),
            "samples": [{"at": fmt_clock(p["start"]), "text": p["text"][:300]} for p in
                        sorted(ps[:3], key=lambda p: p["start"])],
        }
    review = {"diarization_created": diar.get("created"), "method": diar.get("method"),
              "speakers": speakers}
    write_json(job.speakers_path, review)
    stills = [(job.dir / s["still"], f"{k}  {s['share'] * 100:.0f}%") for k, s in speakers.items() if s["still"]]
    if stills:
        contact_sheet(stills, job.dir / "speakers" / "_speakers.jpg", cols=min(4, len(stills)))
    return review


def print_speakers(job: Job) -> None:
    if not job.speakers_path.exists():
        print("no diarization for this job (single speaker / --diarizer none)")
        return
    sp = read_json(job.speakers_path)
    d = read_json(job.diar_path) if job.diar_path.exists() else {}
    est = " (count estimated — confirm with the user)" if d.get("estimated") else ""
    print(f"speakers: {len(sp['speakers'])}{est}   method: {sp.get('method')}")
    if (job.dir / "speakers" / "_speakers.jpg").exists():
        print(f"stills of each speaker talking: {job.dir / 'speakers' / '_speakers.jpg'}")
    for k, s in sp["speakers"].items():
        print(f"\n{k}  name: {s['name'] or '(unnamed)'}  talk {s['talk_time_sec'] / 60:.1f} min "
              f"({s['share'] * 100:.0f}%), first heard {s['first_heard']}")
        if s["still"]:
            print(f"  still: {job.dir / s['still']}  (at {s['still_time']})")
        for q in s["samples"]:
            print(f"  [{q['at']}] {q['text'][:200]}")


def _render(job: Job, args) -> list[Path]:
    from .render import build_words, cues, make_fixer, paragraphs, to_srt, to_transcript, to_vtt

    asr = read_json(job.asr_path)
    diar = read_json(job.diar_path) if job.diar_path.exists() else None
    names = _names(job)
    words = build_words(asr, diar, names)
    if getattr(args, "no_speakers", False):
        for w in words:
            w["speaker"] = None
    fix = make_fixer(job.load().get("replacements"))
    cs = cues(words, max_chars=args.max_chars, max_lines=args.max_lines,
              speaker_tags=args.srt_speakers, fix=fix)
    base = job.base
    job.out.mkdir(parents=True, exist_ok=True)
    files = {
        job.out / f"{base}.srt": to_srt(cs),
        job.out / f"{base}.vtt": to_vtt(cs),
        job.out / f"{base}.transcript.txt": to_transcript(paragraphs(words, fix=fix)),
    }
    for p, txt in files.items():
        p.write_text(txt, encoding="utf-8")
    write_json(job.out / f"{base}.words.json", [
        {k: w[k] for k in ("start", "end", "word", "prob", "speaker")} for w in words])
    unnamed = [k for k in (sorted({w["speaker_id"] for w in words if w["speaker_id"]}))
               if k not in names]
    job.update(rendered=now_iso())
    log(f"rendered {len(cs)} subtitle cues; outputs in {job.out}")
    if unnamed and diar:
        log(f"unnamed speakers: {', '.join(unnamed)} — run `vp speakers {job.name}` then `vp name`")
    return list(files)


# ---------------------------------------------------------------- commands

def cmd_probe(args):
    print(json.dumps(media.summarize(media.probe(Path(args.video))), indent=2))


def cmd_transcribe(args):
    from .transcribe import transcribe

    job = resolve_job(args.video, create=True, job_name=args.job)
    log(f"job {job.name}  ({job.dir})")
    meta = _audio_step(job, args)
    m = meta["media"]
    log(f"source: {m['duration'] / 60:.1f} min, video {m['video'][0] if m['video'] else 'none'}")
    if not job.asr_path.exists() or args.force:
        transcribe(job.audio, job.asr_path, job.asr_partial, model=args.model,
                   language=args.language, prompt=args.prompt, hotwords=args.hotwords,
                   beam_size=args.beam, condition_on_previous=not args.no_condition,
                   threads=args.threads)
        job.update(asr={"model": model_id(args.model), "language": args.language,
                        "prompt": args.prompt, "hotwords": args.hotwords, "at": now_iso()})
    else:
        log("transcript exists (asr.json); pass --force to redo")
    if args.speakers == 1:
        args.diarizer = "none"
    _run_diarize(job, args)
    _render(job, args)
    print_speakers(job)
    print(f"\njob: {job.name}\noutputs: {job.out}")


def cmd_diarize(args):
    job = resolve_job(args.job)
    if not job.asr_path.exists():
        die("no transcript yet; run `vp transcribe` first")
    _run_diarize(job, args)
    _render(job, args)
    print_speakers(job)


def cmd_speakers(args):
    print_speakers(resolve_job(args.job))


def cmd_name(args):
    job = resolve_job(args.job)
    if not job.speakers_path.exists():
        die("no speakers to name (no diarization)")
    sp = read_json(job.speakers_path)
    for a in args.assignments:
        if "=" not in a:
            die(f"expected SPEAKER_XX=Name, got {a!r}")
        k, v = a.split("=", 1)
        k = k.strip().upper()
        if k.isdigit():
            k = f"SPEAKER_{int(k):02d}"
        if k not in sp["speakers"]:
            die(f"unknown speaker {k}; have {', '.join(sp['speakers'])}")
        sp["speakers"][k]["name"] = v.strip() or None
    write_json(job.speakers_path, sp)
    for k, s in sp["speakers"].items():
        print(f"{k} → {s['name'] or '(unnamed)'}")
    _render(job, args)


def cmd_fix(args):
    job = resolve_job(args.job)
    meta = job.load()
    reps = meta.get("replacements", [])
    if args.clear:
        reps = []
    for a in args.replacements:
        if "=>" not in a:
            die(f"expected 'wrong=>right', got {a!r}")
        wrong, right = (x.strip() for x in a.split("=>", 1))
        reps = [r for r in reps if r[0] != wrong] + [[wrong, right]]
    job.update(replacements=reps)
    for wrong, right in reps:
        print(f"  {wrong!r} → {right!r}")
    if job.asr_path.exists():
        _render(job, args)


def cmd_render(args):
    job = resolve_job(args.job)
    for p in _render(job, args):
        print(p)


def cmd_stills(args):
    from .stills import make_stills

    job = resolve_job(args.target, create=True, job_name=args.job)
    meta = job.load()
    if "media" not in meta:
        meta = job.update(media=media.summarize(media.probe(job.source)))
    if not meta["media"]["video"]:
        die("source has no video stream")
    at = [parse_ts(t) for t in args.at.split(",")] if args.at else None
    out = job.stills / args.set
    idx = make_stills(job.source, out, meta["media"]["duration"], at=at,
                      every=parse_ts(args.every) if args.every else None,
                      best=args.best, scenes=args.scenes, threshold=args.threshold,
                      max_scenes=args.max_scenes, ext="png" if args.png else "jpg",
                      width=args.width)
    print(f"{len(idx['stills'])} stills in {out}")
    print(f"contact sheet: {out / idx['contact_sheet']}")
    for s in idx["stills"]:
        print(f"  {s['clock']}  {s['why']:6s} {s['file']}")


def cmd_jobs(args):
    if not WORK.exists():
        return
    for d in sorted(WORK.iterdir()):
        j = Job(d.name)
        if not j.exists():
            continue
        m = j.load()
        state = []
        if j.asr_partial.exists() and not j.asr_path.exists():
            state.append("transcribing (or interrupted)")
        if j.asr_path.exists():
            state.append("transcribed")
        if j.speakers_path.exists():
            sp = read_json(j.speakers_path)["speakers"]
            named = sum(1 for s in sp.values() if s.get("name"))
            state.append(f"{len(sp)} speakers ({named} named)")
        if m.get("rendered"):
            state.append("rendered")
        dur = m.get("media", {}).get("duration")
        print(f"{j.name:50s} {fmt_clock(dur) if dur else '':>9s}  {', '.join(state) or 'new'}")


def cmd_models(args):
    from .transcribe import load_model

    for m in args.model:
        load_model(m)
    if not args.no_diarizer:
        import numpy as np
        from .diarize import embed_windows
        log("fetching ECAPA speaker-embedding model")
        embed_windows(np.zeros(16000 * 2, dtype=np.float32), [(0.0, 1.5)])
    log("models ready")


# ---------------------------------------------------------------- parser

def _add_diar(p, default_diarizer=True):
    g = p.add_argument_group("speakers")
    g.add_argument("--speakers", type=int, help="exact number of speakers, if known (1 = skip diarization)")
    g.add_argument("--min-speakers", type=int)
    g.add_argument("--max-speakers", type=int)
    g.add_argument("--diarizer", choices=["ecapa", "pyannote", "none"], default="ecapa")


def _add_render(p):
    g = p.add_argument_group("output")
    g.add_argument("--max-chars", type=int, default=42, help="subtitle characters per line (default 42)")
    g.add_argument("--max-lines", type=int, default=2, help="subtitle lines per cue (default 2)")
    g.add_argument("--srt-speakers", action="store_true",
                   help="prefix subtitle cues with [Speaker] when the speaker changes")
    g.add_argument("--no-speakers", action="store_true", help="omit speaker labels everywhere")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vp", description="Video prep: transcripts, subtitles, stills.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="show a video's streams and duration")
    p.add_argument("video")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("transcribe", help="audio → transcript → speakers → SRT/VTT/TXT")
    p.add_argument("video")
    p.add_argument("--job", help="job name (default: slug of the file name)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="whisper model: large-v3-turbo (default), large-v3, distil-large-v3, medium.en, small.en …")
    p.add_argument("--language", default="en", help="language code, or 'auto' to detect")
    p.add_argument("--prompt", help="context / vocabulary prompt (names, jargon, acronyms)")
    p.add_argument("--hotwords", help="comma/space separated terms to bias toward")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--no-condition", action="store_true",
                   help="don't condition on previous text (use if you see repetition loops)")
    p.add_argument("--threads", type=int)
    p.add_argument("--audio-stream", type=int, default=0, help="which audio stream (0-based)")
    p.add_argument("--channel", help="use one channel only: L, R or an index (default: downmix)")
    p.add_argument("--force", action="store_true", help="redo audio extraction and transcription")
    _add_diar(p)
    _add_render(p)
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("diarize", help="re-run speaker detection on an existing transcript")
    p.add_argument("job")
    _add_diar(p)
    _add_render(p)
    p.set_defaults(fn=cmd_diarize)

    p = sub.add_parser("speakers", help="show detected speakers, sample quotes and stills")
    p.add_argument("job")
    p.set_defaults(fn=cmd_speakers)

    p = sub.add_parser("name", help="name speakers: SPEAKER_00='Dr. Smith' (same name twice = merge)")
    p.add_argument("job")
    p.add_argument("assignments", nargs="+")
    _add_render(p)
    p.set_defaults(fn=cmd_name)

    p = sub.add_parser("fix", help="whole-word transcript corrections: 'nidle decompression=>needle decompression'")
    p.add_argument("job")
    p.add_argument("replacements", nargs="*")
    p.add_argument("--clear", action="store_true", help="drop all stored corrections first")
    _add_render(p)
    p.set_defaults(fn=cmd_fix)

    p = sub.add_parser("render", help="re-write SRT/VTT/TXT from the stored transcript")
    p.add_argument("job")
    _add_render(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("stills", help="pull still frames (thumbnails, graphics)")
    p.add_argument("target", help="job name or video file")
    p.add_argument("--job", help="job name when target is a file")
    p.add_argument("--at", help="comma-separated timestamps, e.g. 1:23,4:56.5,1h2m3s")
    p.add_argument("--every", help="one still per interval, e.g. 30s or 2m")
    p.add_argument("--best", type=int, help="the N sharpest, best-exposed frames, spread across the video")
    p.add_argument("--scenes", action="store_true", help="a still after each scene cut (full decode: slow)")
    p.add_argument("--threshold", type=float, default=0.3, help="scene-cut sensitivity (lower = more cuts)")
    p.add_argument("--max-scenes", type=int, default=60)
    p.add_argument("--set", default="default", help="sub-folder name for this batch of stills")
    p.add_argument("--png", action="store_true", help="lossless PNG instead of high-quality JPEG")
    p.add_argument("--width", type=int, help="scale to this width (default: full resolution)")
    p.set_defaults(fn=cmd_stills)

    p = sub.add_parser("jobs", help="list jobs in work/")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("models", help="download/warm models into ./models")
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL])
    p.add_argument("--no-diarizer", action="store_true")
    p.set_defaults(fn=cmd_models)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "language", None) == "auto":
        args.language = None
    if getattr(args, "max_speakers", None) and getattr(args, "speakers", None):
        args.min_speakers = args.max_speakers = None
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
