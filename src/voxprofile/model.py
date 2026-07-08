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
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

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

# Optional, additive function/tool-call events. A turn may carry 0..N of these;
# they are paired (start -> result) and rendered as extra waterfall rows. They
# never affect the four boundary stages, so turns without them are unchanged.
FUNC_START_EVENT = "function_call_start"
FUNC_RESULT_EVENT = "function_call_result"
_FUNC_EVENTS = {FUNC_START_EVENT, FUNC_RESULT_EVENT}


def _warn(msg: str) -> None:
    print(f"voxprofile: {msg}", file=sys.stderr)


@dataclass
class FunctionCall:
    """A single function/tool invocation inside a turn.

    Timestamps are epoch seconds (the same clock as boundary events), so a call
    can be placed on the turn timeline via its offset from the turn start. An
    unfinished call (start seen, no result) keeps ``result_t=None``.
    """

    name: str
    start_t: float                      # epoch seconds when the call began
    result_t: Optional[float] = None    # epoch seconds when the result returned
    call_id: Optional[str] = None       # pipecat tool_call_id, used for pairing

    @property
    def duration(self) -> Optional[float]:
        """Execution latency in ms, or None if the call never returned."""
        if self.result_t is None:
            return None
        return (self.result_t - self.start_t) * 1000.0

    @property
    def finished(self) -> bool:
        return self.result_t is not None


@dataclass
class Turn:
    """A single conversational turn with derived latency stages (ms)."""

    turn_id: int
    stages: dict[str, float]  # label -> duration in ms
    total: float              # total latency in ms
    source: str = ""          # originating file, for cross-run stats
    calls: list[FunctionCall] = field(default_factory=list)  # tool calls, if any
    t0: float = 0.0           # epoch seconds of the turn start (timeline origin)

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
    # key -> list of raw (kind, t, name, call_id) function-call records, in
    # file order. Kept separate from ``grouped`` so tool calls never masquerade
    # as boundary events or trip the duplicate-event guard.
    raw_calls: dict[str, list[tuple[str, float, str, Optional[str]]]] = {}

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
        if event in _FUNC_EVENTS:
            name = rec.get("name")
            name = str(name) if isinstance(name, (str, int, float)) else ""
            if not name:
                name = "?"
            call_id = rec.get("call_id")
            call_id = str(call_id) if isinstance(call_id, (str, int, float)) else None
            kind = "start" if event == FUNC_START_EVENT else "result"
            raw_calls.setdefault(key, []).append((kind, t, name, call_id))
            continue
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
        calls = _pair_calls(raw_calls.get(key, []), path, turn_id, warn)
        turns.append(
            Turn(
                turn_id=turn_id,
                stages=stages,
                total=total,
                source=path,
                calls=calls,
                t0=events[_TOTAL_START],
            )
        )

    return turns


def _pair_calls(
    records: list[tuple[str, float, str, Optional[str]]],
    path: str,
    turn_id: object,
    warn: Callable[[str], None],
) -> list[FunctionCall]:
    """Pair function_call_start/result records into :class:`FunctionCall`\\ s.

    Pairing prefers ``call_id`` (pipecat ``tool_call_id``); records without an
    id fall back to FIFO pairing in timestamp order. A start with no matching
    result becomes an unfinished call (``result_t=None``); a result with no
    matching start is a malformed orphan and is dropped with a warning.
    """
    if not records:
        return []

    starts = [(t, name, cid) for kind, t, name, cid in records if kind == "start"]
    results = [(t, name, cid) for kind, t, name, cid in records if kind == "result"]

    # Bucket results that carry an id for exact matching; the rest go FIFO.
    results_by_id: dict[str, list[tuple[float, str]]] = {}
    results_no_id: list[tuple[float, str]] = []
    for t, name, cid in results:
        if cid is not None:
            results_by_id.setdefault(cid, []).append((t, name))
        else:
            results_no_id.append((t, name))
    for bucket in results_by_id.values():
        bucket.sort(key=lambda x: x[0])
    results_no_id.sort(key=lambda x: x[0])

    calls: list[FunctionCall] = []
    starts_no_id: list[tuple[float, str]] = []
    for t, name, cid in starts:
        if cid is not None:
            bucket = results_by_id.get(cid)
            if bucket:
                rt, rname = bucket.pop(0)
                calls.append(
                    FunctionCall(name=name or rname, start_t=t, result_t=rt, call_id=cid)
                )
            else:
                calls.append(FunctionCall(name=name, start_t=t, result_t=None, call_id=cid))
        else:
            starts_no_id.append((t, name))

    starts_no_id.sort(key=lambda x: x[0])
    ri = 0
    for t, name in starts_no_id:
        if ri < len(results_no_id):
            rt, _ = results_no_id[ri]
            ri += 1
            calls.append(FunctionCall(name=name, start_t=t, result_t=rt, call_id=None))
        else:
            calls.append(FunctionCall(name=name, start_t=t, result_t=None, call_id=None))

    orphans = sum(len(b) for b in results_by_id.values()) + max(
        0, len(results_no_id) - ri
    )
    if orphans:
        warn(
            f"{path}: turn {turn_id} has {orphans} function_call_result(s) "
            f"without a matching start"
        )

    calls.sort(key=lambda c: c.start_t)
    return calls


def load_turns_multi(
    paths: Iterable[str], warn: Callable[[str], None] = _warn
) -> list[Turn]:
    """Parse and concatenate turns from several JSONL files."""
    result: list[Turn] = []
    for path in paths:
        result.extend(load_turns(path, warn=warn))
    return result
