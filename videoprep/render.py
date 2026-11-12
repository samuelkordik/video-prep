"""Render words (+ speakers) into SRT / VTT subtitles and a timed transcript."""
from __future__ import annotations

import re

from .common import fmt_clock, fmt_srt, fmt_vtt

# don't leave these dangling at the end of a cue when splitting mid-sentence
WEAK_END = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
            "by", "from", "as", "that", "is", "are", "was", "we", "i", "you", "it", "if", "so"}
SENT_END = re.compile(r"[.?!…]['\")\]]*$")
CLAUSE_END = re.compile(r"[,;:—–-]['\")\]]*$")


def build_words(asr: dict, diar: dict | None, names: dict[str, str] | None) -> list[dict]:
    from .diarize import flatten_words

    words = flatten_words(asr)
    labels = diar["labels"] if diar else [None] * len(words)
    if diar and len(labels) != len(words):
        raise SystemExit("diarization is out of date with the transcript; re-run `vp diarize`")
    names = names or {}
    for w, l in zip(words, labels):
        w["speaker_id"] = l
        w["speaker"] = names.get(l) or l if l else None
    return words


def _join(ws: list[dict]) -> str:
    return re.sub(r"\s+", " ", "".join(w["word"] for w in ws)).strip()


# ---------------------------------------------------------------- subtitles

def _split_lines(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    best, best_score = None, 1e9
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > max_chars or len(b) > max_chars:
            continue
        score = abs(len(a) - len(b))
        if CLAUSE_END.search(words[i - 1]) or SENT_END.search(words[i - 1]):
            score -= 8  # prefer breaking after punctuation
        if score < best_score:
            best, best_score = [a, b], score
    return best or [text]


def make_fixer(replacements: list[list[str]] | None):
    """Whole-word text corrections applied at render time: [["nidle", "needle"], …]."""
    pats = [(re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)"), b) for a, b in (replacements or [])]

    def fix(text: str) -> str:
        for rx, b in pats:
            text = rx.sub(b, text)
        return text
    return fix


def cues(words: list[dict], max_chars: int = 42, max_lines: int = 2, max_dur: float = 7.0,
         min_dur: float = 1.0, gap_break: float = 1.0, speaker_tags: bool = False,
         fix=lambda t: t) -> list[dict]:
    limit = max_chars * max_lines
    out: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                        "words": list(cur), "speaker": cur[0].get("speaker")})
            cur.clear()

    for w in words:
        if cur:
            text_len = len(_join(cur + [w]))
            prev = cur[-1]
            if w.get("speaker_id") != prev.get("speaker_id") or w["start"] - prev["end"] > gap_break:
                flush()
            elif text_len > limit or w["end"] - cur[0]["start"] > max_dur:
                # back off to the last punctuation break if that leaves a decent cue
                for j in range(len(cur) - 1, 0, -1):
                    pw = cur[j - 1]["word"].strip()
                    if (SENT_END.search(pw) or CLAUSE_END.search(pw)) and len(_join(cur[:j])) >= limit * 0.4:
                        rest = cur[j:]
                        del cur[j:]
                        flush()
                        cur.extend(rest)
                        break
                if cur and (len(_join(cur + [w])) > limit or w["end"] - cur[0]["start"] > max_dur):
                    carry = []
                    while len(cur) > 3 and cur[-1]["word"].strip().lower() in WEAK_END and len(carry) < 2:
                        carry.insert(0, cur.pop())
                    flush()
                    cur.extend(carry)
        cur.append(w)
        if SENT_END.search(w["word"].strip()) and len(_join(cur)) >= limit * 0.6:
            flush()
    flush()

    # timing: minimum duration, no overlaps, small tail so text doesn't vanish on the last word
    for i, c in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else c["end"] + 2.0
        end = max(c["end"] + 0.25, c["start"] + min_dur)
        c["end"] = min(end, nxt - 0.04) if nxt - 0.04 > c["start"] else c["end"]

    prev_spk = None
    for c in out:
        text = fix(_join(c["words"]))
        if speaker_tags and c["speaker"] and c["speaker"] != prev_spk:
            text = f"[{c['speaker']}] {text}"
        prev_spk = c["speaker"]
        c["lines"] = _split_lines(text, max_chars)
        del c["words"]
    return out


def to_srt(cs: list[dict]) -> str:
    return "\n".join(f"{i}\n{fmt_srt(c['start'])} --> {fmt_srt(c['end'])}\n" + "\n".join(c["lines"]) + "\n"
                     for i, c in enumerate(cs, 1))


def to_vtt(cs: list[dict]) -> str:
    body = "\n".join(f"{fmt_vtt(c['start'])} --> {fmt_vtt(c['end'])}\n" + "\n".join(c["lines"]) + "\n"
                     for c in cs)
    return "WEBVTT\n\n" + body


# ---------------------------------------------------------------- transcript

def paragraphs(words: list[dict], max_para: float = 60.0, pause_break: float = 2.0,
               fix=lambda t: t) -> list[dict]:
    paras: list[dict] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            prev = cur[-1]
            dur = prev["end"] - cur[0]["start"]
            ended = SENT_END.search(prev["word"].strip())
            if (w.get("speaker_id") != prev.get("speaker_id")
                    or (ended and dur >= max_para)
                    or (ended and w["start"] - prev["end"] >= pause_break and dur >= 15)
                    or dur >= max_para * 2):
                paras.append({"start": cur[0]["start"], "end": prev["end"],
                              "speaker": cur[0].get("speaker"), "text": fix(_join(cur))})
                cur = []
        cur.append(w)
    if cur:
        paras.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                      "speaker": cur[0].get("speaker"), "text": fix(_join(cur))})
    return paras


def to_transcript(paras: list[dict], title: str | None = None) -> str:
    lines = [f"{title}\n"] if title else []
    for p in paras:
        who = f"{p['speaker']}: " if p["speaker"] else ""
        lines.append(f"[{fmt_clock(p['start'])}] {who}{p['text']}\n")
    return "\n".join(lines)
