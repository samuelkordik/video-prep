# video-prep

A small Docker-packaged command-line tool for video production chores:

- **Transcribe** a video with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (Whisper large-v3-turbo by default, CPU int8, word-level timestamps).
- **Find who is speaking** (speaker diarization) and let you name the speakers.
- **Write subtitles** (`.srt` for YouTube upload / NLE import, plus `.vtt`), following
  common caption rules: at most 2 lines × 42 characters, at most 7 s per cue, breaks at
  punctuation, new cue on each speaker change.
- **Write a timed transcript** (`.transcript.txt`) with speaker labels:
  `[00:01:12] Dr. Jane Smith: …`
- **Pull still frames** for thumbnails and graphics: at given times, at intervals, the
  N best-looking (sharp, well-exposed) frames, or after each scene cut. Each batch
  gets a contact sheet.

It is designed to be driven by an AI assistant (Claude Code) through fixed commands —
see [`CLAUDE.md`](CLAUDE.md) for that workflow — but works fine by hand.

Runs on CPU only; no GPU needed.

## Setup

```bash
git clone https://github.com/samuelkordik/video-prep && cd video-prep
docker compose build
./vp models            # download Whisper + speaker model into ./models (~1.7 GB)
```

## Usage

Put videos in `src/` (any format ffmpeg reads: `.mov`, `.mp4`, `.mkv`, ProRes, H.264, HEVC…).

```bash
./vp probe "src/talk.mov"                         # duration, streams
./vp transcribe "src/talk.mov" --speakers 2 \
     --prompt "Needle decompression, tension pneumothorax, ROSC, OMD."
./vp speakers talk                                # who's who: quotes + a still of each
./vp name talk SPEAKER_00="Dr. Jane Smith" SPEAKER_01="John Doe"
./vp stills talk --best 12                        # thumbnail candidates
./vp stills talk --at 1:23,14:05 --png --set title-card
./vp jobs
```

Outputs land in `work/<job>/`:

| Path | What |
|---|---|
| `out/<name>.srt`, `out/<name>.vtt` | subtitles |
| `out/<name>.transcript.txt` | timed transcript with speaker names |
| `out/<name>.words.json` | every word with start/end/speaker/confidence |
| `speakers.json`, `speakers/` | detected speakers, sample quotes, one still each |
| `stills/<set>/` | stills, `_contact.jpg` sheet, `_index.json` |
| `asr.json`, `diarization.json`, `audio.wav` | intermediate results (re-used on re-runs) |

### Options worth knowing

- `--speakers N` if you know the count (much more reliable than auto-detection);
  `--speakers 1` skips diarization. `--min-speakers` / `--max-speakers` bound the estimate.
- `--prompt` gives Whisper context and vocabulary (names, acronyms, jargon).
- `--channel L|R` transcribes one channel only (e.g. two lav mics on separate channels);
  `--audio-stream N` picks another audio track.
- `--srt-speakers` adds `[Name]` to subtitle cues at speaker changes (off by default,
  since YouTube captions usually don't want it).
- `--model`: `large-v3-turbo` (default, best speed/quality), `large-v3` (slower, a little
  more accurate), `distil-large-v3` (English, faster), `medium.en`, `small.en`.
- `vp diarize <job> --speakers 3` re-runs speaker detection without re-transcribing.
- `VP_NOTIFY=1 ./vp …` sends a homelab notification when the command finishes (only
  if the notify script used on the author's server exists).

### Speaker detection

The default `ecapa` diarizer needs no account: it embeds 1.5 s windows of speech with
SpeechBrain's ECAPA-TDNN model, clusters them (spectral clustering; the eigengap picks
a first count, then groups whose two halves sound like different people are split and
groups that sound alike are merged, until the requested count or the thresholds are
met), caches the embeddings so re-runs are quick, and gives each word the speaker of the windows
around it. For harder audio (lots of crosstalk), `--diarizer pyannote` runs
`pyannote/speaker-diarization-3.1`; it needs a Hugging Face token in `.env`
(`HF_TOKEN=…`) and the model's terms accepted on huggingface.co.

## Layout

```
vp                 wrapper: runs the CLI in the container
compose.yaml       mounts src/ (read-only), work/, models/, and the code
Dockerfile         python:3.11-slim + ffmpeg + CPU PyTorch
videoprep/         the CLI (cli.py) and its parts: media, transcribe, diarize, render, stills
src/  work/  models/   gitignored
```
