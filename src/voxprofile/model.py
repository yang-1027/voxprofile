"""Event parsing and per-turn latency stage derivation.

Input is a JSONL stream of events, one JSON object per line::

    {"turn_id": 1, "event": "user_stopped_speaking", "t": 1751600000.000}

``t`` is an epoch timestamp in seconds (float). Events are grouped by
``turn_id`` and turned into four latency stages plus a total, all in
milliseconds.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

# Stage definition: (label, start_event, end_event).
# Each stage duration = t[end_event] - t[start_event].
STAGES: list[tuple[str, str, str]] = [
    ("STT", "user_stopped_speaking", "stt_final"),
    ("LLM", "stt_final", "llm_first_token"),
    ("TTS", "llm_first_token", "tts_first_byte"),
    ("Playback", "tts_first_byte", "playback_started"),
]

STAGE_LABELS: list[str] = [label for label, _, _ in STAGES]

# Every event needed to derive a full turn.
REQUIRED_EVENTS: list[str] = [
    "user_stopped_speaking",
    "stt_final",
    "llm_first_token",
    "tts_first_byte",
    "playback_started",
]

_TOTAL_START = "user_stopped_speaking"
_TOTAL_END = "playback_started"


def _warn(msg: str) -> None:
    print(f"voxprofile: {msg}", file=sys.stderr)


@dataclass
class Turn:
    """A single conversational turn with derived latency stages (ms)."""

    turn_id: int
    stages: dict[str, float]  # label -> duration in ms
    total: float              # total latency in ms
    source: str = ""          # originating file, for cross-run stats

    @property
    def bottleneck(self) -> str:
        """Label of the slowest stage in this turn."""
        return max(self.stages, key=lambda k: self.stages[k])


def _iter_records(path: str) -> Iterable[tuple[int, dict]]:
    """Yield (line_number, record) for each valid JSON line in a file."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                _warn(f"{path}:{lineno}: skipping malformed JSON ({exc.msg})")


def load_turns(path: str, warn: Callable[[str], None] = _warn) -> list[Turn]:
    """Parse one JSONL file into a list of fully-formed :class:`Turn`.

    Turns missing any required event are skipped with a warning on stderr;
    the function never raises for malformed input.
    """
    # normalized str(turn_id) -> {event: t}, preserving first-seen turn order.
    # A normalized key lets ``1`` and ``"1"`` refer to the same turn and keeps
    # the key hashable even for exotic turn_id values.
    grouped: dict[str, dict[str, float]] = {}
    display: dict[str, object] = {}  # key -> first-seen original turn_id
    order: list[str] = []

    for lineno, rec in _iter_records(path):
        if not isinstance(rec, dict):
            warn(f"{path}:{lineno}: skipping non-object record")
            continue
        turn_id = rec.get("turn_id")
        event = rec.get("event")
        t = rec.get("t")
        if turn_id is None or event is None or t is None:
            warn(f"{path}:{lineno}: skipping record missing turn_id/event/t")
            continue
        if not isinstance(turn_id, (int, float, str)):
            warn(f"{path}:{lineno}: skipping record with non-scalar turn_id")
            continue
        try:
            t = float(t)
        except (TypeError, ValueError):
            warn(f"{path}:{lineno}: skipping record with non-numeric t")
            continue
        key = str(turn_id)
        event = str(event)
        if key not in grouped:
            grouped[key] = {}
            display[key] = turn_id
            order.append(key)
        elif event in grouped[key]:
            warn(
                f"{path}:{lineno}: turn {display[key]} has duplicate "
                f"'{event}' event, keeping the latest"
            )
        grouped[key][event] = t

    turns: list[Turn] = []
    for key in order:
        events = grouped[key]
        turn_id = display[key]
        missing = [e for e in REQUIRED_EVENTS if e not in events]
        if missing:
            warn(
                f"{path}: skipping turn {turn_id}, "
                f"missing event(s): {', '.join(missing)}"
            )
            continue
        stages = {
            label: (events[end] - events[start]) * 1000.0
            for label, start, end in STAGES
        }
        total = (events[_TOTAL_END] - events[_TOTAL_START]) * 1000.0
        if any(v < 0 for v in stages.values()):
            warn(f"{path}: turn {turn_id} has out-of-order timestamps")
        turns.append(Turn(turn_id=turn_id, stages=stages, total=total, source=path))

    return turns


def load_turns_multi(
    paths: Iterable[str], warn: Callable[[str], None] = _warn
) -> list[Turn]:
    """Parse and concatenate turns from several JSONL files."""
    result: list[Turn] = []
    for path in paths:
        result.extend(load_turns(path, warn=warn))
    return result
