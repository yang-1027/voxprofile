# voxprofile

A latency waterfall for your voice AI agent.

Voice agents live or die on turn latency, but "why does this turn feel slow" usually means
sprinkling `time.time()` calls across the pipeline and grepping logs. voxprofile records the
five timestamps that matter and turns them into a per-turn waterfall — so you can see exactly
which stage is eating your latency budget, and whether p95 regressed since your last change.

```
Turn 3   total 1290 ms   ❌ over 800 ms target by 490 ms
  STT        ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    210 ms
  LLM        ░░░░░░░█████████████████████████████░░░░░░░░    860 ms  ← bottleneck
  TTS        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████░░░    150 ms
  Playback   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██     70 ms
                                        ┆
                            target 800 ms

Summary  (3 turns)
                   p50       p95
  STT           185 ms    208 ms
  LLM           435 ms    817 ms
  TTS           120 ms    147 ms
  Playback       60 ms     69 ms
  ──────────────────────────────
  Total         800 ms   1241 ms
```

The stage model:

```
user stops speaking → STT final → LLM first token → TTS first byte → playback starts
                 STT          LLM TTFT           TTS TTFB        Playback
```

- **Zero dependencies** — the CLI is pure standard library.
- **Framework-agnostic core** — events are plain JSONL; record them from anywhere.
- **Pipecat observer included** — one object attached to your `PipelineTask`, no pipeline changes.
- **Regression-friendly** — `voxprofile stats` aggregates p50/p95 across runs so latency creep can't hide.

## Install

```bash
pip install git+https://github.com/yang-1027/voxprofile.git

# with the Pipecat observer
pip install "voxprofile[pipecat] @ git+https://github.com/yang-1027/voxprofile.git"
```

## Quickstart (60 seconds, no API keys)

```bash
git clone https://github.com/yang-1027/voxprofile.git
cd voxprofile && pip install -e .

voxprofile replay examples/sample_events.jsonl            # per-turn waterfalls + summary
voxprofile replay examples/sample_events.jsonl --html waterfall.html   # shareable HTML report
voxprofile stats examples/*.jsonl                         # p50/p95 across runs
```

## Record from a Pipecat pipeline

```python
from voxprofile.pipecat_observer import VoxprofileObserver

observer = VoxprofileObserver("latency.jsonl")
task = PipelineTask(pipeline, observers=[observer])
# ... run the pipeline ...
observer.close()
```

Then inspect the recording:

```bash
voxprofile replay latency.jsonl --target 800
```

Verified against Pipecat 1.5.0. Frame mapping: `VADUserStoppedSpeakingFrame` (or
`UserStoppedSpeakingFrame`) → turn start, `TranscriptionFrame` → STT final, first
`LLMTextFrame` → LLM first token, first `TTSAudioRawFrame` → TTS first byte,
`BotStartedSpeakingFrame` → playback started. Interrupted (barge-in) turns are kept as
incomplete and skipped by `replay`. See the module docstring for what is intentionally
not captured.

## Record from anything else

Events are one JSON object per line — emit them from any stack:

```json
{"turn_id": 1, "event": "user_stopped_speaking", "t": 1751700000.000}
{"turn_id": 1, "event": "stt_final",             "t": 1751700000.140}
{"turn_id": 1, "event": "llm_first_token",       "t": 1751700000.460}
{"turn_id": 1, "event": "tts_first_byte",        "t": 1751700000.555}
{"turn_id": 1, "event": "playback_started",      "t": 1751700000.610}
```

`t` is epoch seconds from a single clock. Turns with missing events are reported and skipped,
malformed lines never crash the parser.

## CLI

| Command | What it does |
|---|---|
| `voxprofile replay <events.jsonl>` | Per-turn waterfall + p50/p95 summary |
| `voxprofile replay ... --target 500` | Pass/fail against your own latency budget (default 800 ms) |
| `voxprofile replay ... --html report.html` | Self-contained HTML report (no external assets) |
| `voxprofile replay ... --no-color` | Plain output for CI logs |
| `voxprofile stats <run1.jsonl> <run2.jsonl> ...` | p50/p95/min/max per stage across runs |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
