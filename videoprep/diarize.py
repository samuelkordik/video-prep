"""Speaker diarization on top of the ASR word timings.

Two backends:
  ecapa    (default, no account needed) SpeechBrain ECAPA-TDNN speaker
           embeddings over sliding windows of the speech, spectral clustering
           (eigengap picks the speaker count when it isn't given), then each
           word gets the speaker of the windows that cover it.
  pyannote (needs HF_TOKEN and accepting the pyannote/speaker-diarization-3.1
           model terms on Hugging Face) the full pyannote pipeline; words get
           the speaker turn they overlap most.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from .common import MODELS, log

WIN = 1.5     # embedding window (s)
HOP = 0.75    # window hop (s)
SR = 16000


def flatten_words(asr: dict) -> list[dict]:
    words = []
    for s in asr["segments"]:
        for w in s.get("words", []):
            if w["word"].strip():
                words.append({**w, "seg": s["id"]})
    return words


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- windows

def speech_windows(words: list[dict], total: float) -> list[tuple[float, float]]:
    regions: list[list[float]] = []
    for w in words:
        if regions and w["start"] - regions[-1][1] <= 0.6:
            regions[-1][1] = max(regions[-1][1], w["end"])
        else:
            regions.append([w["start"], w["end"]])
    wins = []
    for a, b in regions:
        if b - a < 0.3:
            continue
        if b - a < WIN:
            c = (a + b) / 2
            s = min(max(0.0, c - WIN / 2), max(0.0, total - WIN))
            wins.append((s, s + WIN))
            continue
        t = a
        while t + WIN <= b:
            wins.append((t, t + WIN))
            t += HOP
        if wins[-1][1] < b - 0.3:
            wins.append((b - WIN, b))
    return wins


def embed_windows(audio: np.ndarray, wins: list[tuple[float, float]], batch: int = 64) -> np.ndarray:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS / "speechbrain" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )
    n = int(WIN * SR)
    out = []
    with torch.inference_mode():
        for i in range(0, len(wins), batch):
            chunk = []
            for s, _ in wins[i:i + batch]:
                a = int(s * SR)
                x = audio[a:a + n]
                if len(x) < n:
                    x = np.pad(x, (0, n - len(x)))
                chunk.append(x)
            e = enc.encode_batch(torch.from_numpy(np.stack(chunk)))
            out.append(e.squeeze(1).numpy())
            if (i // batch) % 20 == 0:
                log(f"embeddings {min(i + batch, len(wins))}/{len(wins)}")
    return np.concatenate(out)


# ---------------------------------------------------------------- clustering

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def spectral(X: np.ndarray, k: int | None, max_k: int = 8) -> tuple[np.ndarray, int, list[float]]:
    from sklearn.cluster import KMeans

    n = len(X)
    A = X @ X.T
    np.fill_diagonal(A, 0)
    keep = max(10, int(0.05 * n))
    keep = min(keep, n - 1)
    thr = -np.sort(-A, axis=1)[:, keep - 1:keep]
    A = np.where(A >= thr, np.maximum(A, 0), 0.0)
    A = (A + A.T) / 2
    d = A.sum(1)
    dinv = 1 / np.sqrt(d + 1e-9)
    L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    vals, vecs = np.linalg.eigh(L)
    m = min(max_k + 1, n)
    gaps = list(np.diff(vals[:m]))
    if k is None:
        k = int(np.argmax(gaps[: max_k]) + 1) if len(gaps) else 1
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int), 1, [round(float(g), 4) for g in gaps]
    E = _normalize(vecs[:, :k])
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(E)
    return labels, k, [round(float(g), 4) for g in gaps]


def centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ks = sorted(set(labels.tolist()))
    return _normalize(np.stack([X[labels == c].mean(0) for c in ks])), ks


def cluster(emb: np.ndarray, num_speakers: int | None, min_speakers: int | None,
            max_speakers: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    X = _normalize(emb)
    n = len(X)
    # spectral is O(n^2) memory / O(n^3) time: cluster a subsample of long files
    cap = 4000
    idx = np.arange(n) if n <= cap else np.linspace(0, n - 1, cap).astype(int)
    max_k = max_speakers or 8
    labels_s, k, gaps = spectral(X[idx], num_speakers, max_k=max_k)
    if num_speakers is None and min_speakers and k < min_speakers:
        labels_s, k, gaps = spectral(X[idx], min_speakers, max_k=max_k)
    C, _ = centroids(X[idx], labels_s)
    # a few rounds of centroid refinement over all windows
    labels = np.argmax(X @ C.T, axis=1)
    for _ in range(10):
        C, ks = centroids(X, labels)
        new = np.array(ks)[np.argmax(X @ C.T, axis=1)]
        if np.array_equal(new, labels):
            break
        labels = new
    info = {"eigengaps": gaps, "estimated": num_speakers is None}
    if num_speakers is None:
        # fold clusters with <2% of speech into their nearest neighbour
        while True:
            C, ks = centroids(X, labels)
            share = np.array([(labels == c).mean() for c in ks])
            if len(ks) <= max(1, min_speakers or 1) or share.min() >= 0.02:
                break
            small = ks[int(np.argmin(share))]
            mask = labels == small
            others = [i for i, c in enumerate(ks) if c != small]
            labels[mask] = np.array(ks)[others][np.argmax(X[mask] @ C[others].T, axis=1)]
    C, ks = centroids(X, labels)
    remap = {c: i for i, c in enumerate(ks)}
    return np.array([remap[c] for c in labels]), C, info


# ---------------------------------------------------------------- word assignment

def assign_words(words: list[dict], wins: list[tuple[float, float]], X: np.ndarray,
                 C: np.ndarray) -> list[int]:
    X = _normalize(X)
    starts = np.array([w[0] for w in wins])
    ends = np.array([w[1] for w in wins])
    centers = (starts + ends) / 2
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        cover = np.nonzero((starts <= mid) & (ends >= mid))[0]
        if len(cover) == 0:
            cover = [int(np.argmin(np.abs(centers - mid)))]
        v = X[cover].mean(0)
        out.append(int(np.argmax(C @ v)))
    return out


def smooth(words: list[dict], labels: list[int], min_dur: float = 1.0, min_words: int = 4) -> list[int]:
    """Absorb short speaker blips sandwiched between the same speaker."""
    labels = list(labels)
    for _ in range(3):
        runs = []  # [label, i0, i1]
        for i, l in enumerate(labels):
            if runs and runs[-1][0] == l:
                runs[-1][2] = i
            else:
                runs.append([l, i, i])
        changed = False
        for j in range(1, len(runs) - 1):
            l, a, b = runs[j]
            dur = words[b]["end"] - words[a]["start"]
            if runs[j - 1][0] == runs[j + 1][0] != l and dur < min_dur and (b - a + 1) < min_words:
                for i in range(a, b + 1):
                    labels[i] = runs[j - 1][0]
                changed = True
        if not changed:
            break
    return labels


def order_by_first_appearance(labels: list[int]) -> list[int]:
    order: dict[int, int] = {}
    for l in labels:
        order.setdefault(l, len(order))
    return [order[l] for l in labels]


# ---------------------------------------------------------------- backends

def diarize_ecapa(audio_path: Path, words: list[dict], num_speakers=None,
                  min_speakers=None, max_speakers=None) -> dict:
    audio = load_wav(audio_path)
    total = len(audio) / SR
    wins = speech_windows(words, total)
    log(f"diarize (ecapa): {len(wins)} windows over {sum(b - a for a, b in wins) / 60:.1f} min")
    if not wins:
        return {"method": "ecapa", "num_speakers": 1, "labels": [0] * len(words)}
    emb = embed_windows(audio, wins)
    if num_speakers == 1:
        labels = [0] * len(words)
        info = {"estimated": False}
    else:
        _, C, info = cluster(emb, num_speakers, min_speakers, max_speakers)
        labels = smooth(words, assign_words(words, wins, emb, C))
    labels = order_by_first_appearance(labels)
    return {"method": "ecapa", "num_speakers": len(set(labels)), **info, "labels": labels}


def diarize_pyannote(audio_path: Path, words: list[dict], num_speakers=None,
                     min_speakers=None, max_speakers=None) -> dict:
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("pyannote needs HF_TOKEN (and accepted model terms); use --diarizer ecapa")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    pipe.to(torch.device("cpu"))
    audio = torch.from_numpy(load_wav(audio_path))[None, :]
    log("diarize (pyannote): running pipeline (slow on CPU)")
    kw = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                 max_speakers=max_speakers).items() if v}
    ann = pipe({"waveform": audio, "sample_rate": SR}, **kw)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    names = sorted({s for *_, s in turns})
    ix = {s: i for i, s in enumerate(names)}
    labels = []
    for w in words:
        best, best_ov, best_dist = 0, 0.0, 1e9
        for a, b, s in turns:
            ov = min(b, w["end"]) - max(a, w["start"])
            dist = 0 if ov > 0 else min(abs(a - w["end"]), abs(w["start"] - b))
            if ov > best_ov or (best_ov <= 0 and dist < best_dist):
                best, best_ov, best_dist = ix[s], ov, dist
        labels.append(best)
    labels = order_by_first_appearance(smooth(words, labels))
    return {"method": "pyannote", "num_speakers": len(set(labels)), "estimated": not num_speakers,
            "labels": labels}


def diarize(audio_path: Path, asr: dict, method: str = "ecapa", **kw) -> dict:
    words = flatten_words(asr)
    fn = {"ecapa": diarize_ecapa, "pyannote": diarize_pyannote}[method]
    res = fn(audio_path, words, **kw)
    res["labels"] = [f"SPEAKER_{l:02d}" for l in res["labels"]]
    return res
