"""Speech-to-text with faster-whisper (CTranslate2, CPU int8), word timestamps."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .common import MODELS, log, write_json

# Short names accepted on the command line → faster-whisper model ids.
MODEL_ALIASES = {
    "turbo": "large-v3-turbo",
    "large": "large-v3",
    "distil": "distil-large-v3",
}
DEFAULT_MODEL = "large-v3-turbo"


def model_id(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_model(name: str, threads: int | None = None):
    from faster_whisper import WhisperModel

    name = model_id(name)
    threads = threads or os.cpu_count() or 4
    log(f"loading whisper model {name} (cpu int8, {threads} threads)")
    return WhisperModel(name, device="cpu", compute_type="int8",
                        cpu_threads=threads, download_root=str(MODELS / "whisper"))


def transcribe(audio: Path, out: Path, partial: Path, *, model: str = DEFAULT_MODEL,
               language: str | None = "en", prompt: str | None = None,
               hotwords: str | None = None, beam_size: int = 5,
               condition_on_previous: bool = True, threads: int | None = None) -> dict:
    """Transcribe `audio` → `out` (json). Streams segments to `partial` as it goes."""
    wm = load_model(model, threads)
    t0 = time.time()
    segments, info = wm.transcribe(
        str(audio), language=language, initial_prompt=prompt or None,
        hotwords=hotwords or None, beam_size=beam_size, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=condition_on_previous,
    )
    dur = info.duration
    log(f"language={info.language} (p={info.language_probability:.2f}); audio {dur / 60:.1f} min, "
        f"{info.duration_after_vad / 60:.1f} min of speech after VAD")
    segs = []
    last = 0.0
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as pf:
        for s in segments:
            seg = {
                "id": s.id, "start": round(s.start, 3), "end": round(s.end, 3),
                "text": s.text.strip(), "avg_logprob": round(s.avg_logprob, 3),
                "no_speech_prob": round(s.no_speech_prob, 3),
                "words": [{"start": round(w.start, 3), "end": round(w.end, 3),
                           "word": w.word, "prob": round(w.probability, 3)}
                          for w in (s.words or [])],
            }
            segs.append(seg)
            pf.write(json.dumps(seg, ensure_ascii=False) + "\n")
            pf.flush()
            if s.end - last >= 120 or s.end >= dur - 1:
                last = s.end
                el = time.time() - t0
                rate = s.end / el if el else 0
                eta = (dur - s.end) / rate if rate else 0
                log(f"asr {100 * s.end / dur:5.1f}%  at {s.end / 60:5.1f} min  "
                    f"speed {rate:.2f}x realtime  eta {eta / 60:.0f} min")
    result = {
        "model": model_id(model), "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": dur, "prompt": prompt, "hotwords": hotwords,
        "elapsed_sec": round(time.time() - t0, 1), "segments": segs,
    }
    write_json(out, result)
    partial.unlink(missing_ok=True)
    log(f"asr done: {len(segs)} segments in {result['elapsed_sec'] / 60:.1f} min")
    return result
