# video-prep — instructions for Claude

A CLI tool (in Docker) that turns a video into subtitles (SRT/VTT), a timed transcript with
named speakers, and still frames for thumbnails/graphics. **You drive it with the fixed
commands below; don't write ad-hoc ffmpeg/whisper scripts for these jobs.** If a job needs
something the tool can't do, add it to `videoprep/` (code is bind-mounted, so no rebuild
needed unless `requirements.txt` changes) and commit.

Host: trantor (CPU only — i7-4810MQ, 8 threads, no GPU). Homelab deconfliction: no ports,
no hostnames; registered in `~/dev/homelab/REGISTRY.md` as `videoprep`.

## Layout
- `src/` — source videos (user uploads here; gitignored, mounted read-only).
- `work/<job>/` — everything a job produces (gitignored). Outputs in `work/<job>/out/`.
- `models/` — model cache (gitignored). `./vp models` warms it.
- Always run from the repo root via `./vp …`. Quote paths: file names have spaces and `&`.

## Workflow: transcript + subtitles

1. **Probe** the file: `./vp probe "src/<file>"` → duration, audio streams/channels.
   If there are several audio streams or a stereo pair that might be two separate mics,
   mention it; `--audio-stream N` / `--channel L|R` pick one.
2. **Ask the user before transcribing** (one AskUserQuestion, only what you don't already know):
   - How many people speak? (exact count → `--speakers N`; it is far more reliable than
     auto-detection. One speaker → `--speakers 1`, which skips diarization.)
   - Their names, if they want them now (you can also name later from the review).
   - Jargon / names to spell right → `--prompt "…"` (a short natural sentence listing
     terms works best, e.g. `"Needle decompression of tension pneumothorax in traumatic
     cardiac arrest; OMD, ROSC, ETCO2, Dr. Kordik."`).
3. **Transcribe** — long-running, so run it in the background and log to the job:
   ```bash
   VP_NOTIFY=1 ./vp transcribe "src/<file>" --speakers 2 --prompt "…" > work/<slug>.log 2>&1
   ```
   Speed on trantor: see "Measured speed" below. Progress lines (`asr 42.0% … eta`) go to
   the log. The transcript is saved incrementally (`asr.partial.jsonl`), but a restart
   re-transcribes from the start.
4. **Identify speakers**: `./vp speakers <job>`. Look at `work/<job>/speakers/_speakers.jpg`
   (Read it — one still per speaker, taken while that speaker talks) and the sample
   quotes. Match to the names the user gave (introductions in the quotes help), then
   **confirm with the user** unless it's unambiguous. Name them:
   `./vp name <job> SPEAKER_00="Dr. Jane Smith" SPEAKER_01="John Doe"`.
   Giving two IDs the same name merges them (diarization split one person).
   If the count was wrong: `./vp diarize <job> --speakers N` (minutes, no re-transcribe).
5. **Proofread**: skim `out/<name>.transcript.txt` for misheard jargon/names. Fix with
   whole-word replacements (stored in the job, re-applied on every render):
   `./vp fix <job> "nidle decompression=>needle decompression" "Dr. Cordic=>Dr. Kordik"`.
   **Check on-screen lower thirds** (`./vp stills <job> --at <first_heard>+5s …`): they
   give the authoritative spelling of names — in testing Whisper wrote "Farouk Mechri"
   where the lower third reads "Dr. Faroukh Mehkri".
   Spot-check a few low-confidence words in `out/<name>.words.json` (`prob` < 0.5).
6. **Deliver**: tell the user the paths in `work/<job>/out/`:
   `<name>.srt` (YouTube / NLE import), `<name>.vtt`, `<name>.transcript.txt`.
   Subtitles have no speaker tags by default; `./vp render <job> --srt-speakers` adds
   `[Name]` at speaker changes if wanted.

## Workflow: stills for thumbnails / graphics

- Candidates: `./vp stills <job> --best 12` (sharp, well-exposed frames spread over the
  video) → Read `work/<job>/stills/default/_contact.jpg` to see them; propose picks.
- Specific moments: find the time in the transcript (e.g. grep the `.transcript.txt` or
  `words.json` for a phrase), then `./vp stills <job> --at 12:34,12:36.5 --png --set thumb`.
  Grab a few frames ±1 s around a moment; expressions change fast.
- Interval sweep: `--every 30s`. Scene cuts: `--scenes` (decodes the whole file — slow on
  ProRes; tell the user and background it).
- `--set NAME` keeps batches apart; `--png` for lossless (graphics work), JPEG q2 otherwise.
  Full source resolution unless `--width`.

## Rules
- Never commit anything in `src/`, `work/` or `models/` (all gitignored). Public repo:
  no secrets — `HF_TOKEN` goes in `.env` (gitignored) if pyannote is ever used.
- Long commands (transcribe, `--scenes`) go in the background with `VP_NOTIFY=1`.
- Re-running `transcribe` on the same file reuses `asr.json`; `--force` redoes it.
- `vp diarize` resets speaker names (labels can change meaning); re-name afterwards.

## Measured speed (trantor)
- ASR, large-v3-turbo int8, 8 threads: **~1.3× realtime** (4-min test clip in 3.1 min).
  A 24-min video ≈ 18–20 min. Diarization (ECAPA) adds well under a minute per 10 min.
- `stills --best N`: seconds (seeks, no full decode). `--scenes` on ProRes: a full decode.
- Model load ≈ 5 s once cached; first `./vp models` downloads ~1.6 GB.
