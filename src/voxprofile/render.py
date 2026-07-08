"""Terminal (ASCII/Unicode) waterfall rendering."""

from __future__ import annotations

import math
from typing import Sequence

from .model import STAGE_LABELS, FunctionCall, Turn
from .stats import StageStats, aggregate, aggregate_tools

# Chart geometry.
TRACK_WIDTH = 44          # columns for the waterfall track
LABEL_WIDTH = 10          # left column for stage labels
_FILLED = "█"
_TRACK = "░"

# Per-stage ANSI colors (256-color foreground codes).
_STAGE_COLOR = {
    "STT": 44,        # cyan
    "LLM": 214,       # orange
    "TTS": 78,        # green
    "Playback": 105,  # violet
}
_TOOL_COLOR = 173     # coral, distinct from the four stage hues
_DIM = 240
_RESET = "\033[0m"


class Palette:
    """Wraps ANSI coloring; a no-op when ``enabled`` is False."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def fg(self, text: str, code: int) -> str:
        if not self.enabled:
            return text
        return f"\033[38;5;{code}m{text}{_RESET}"

    def bold(self, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[1m{text}{_RESET}"

    def dim(self, text: str) -> str:
        return self.fg(text, _DIM)


def _fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms"


def _scale(global_max: float) -> float:
    return TRACK_WIDTH / global_max if global_max > 0 else 0.0


def _draw_track(
    offset_ms: float, duration: float, scale: float, color: int, pal: Palette
) -> str:
    """Render one TRACK_WIDTH-wide bar at a given offset (shared by all rows)."""
    # Clamp against out-of-order timestamps: a negative cumulative offset or an
    # over-wide bar must never push the track past TRACK_WIDTH.
    start = min(max(round(offset_ms * scale), 0), TRACK_WIDTH)
    width = max(1, round(duration * scale)) if duration > 0 else 0
    width = max(0, min(width, TRACK_WIDTH - start))

    track_before = pal.dim(_TRACK * start)
    bar = pal.fg(_FILLED * width, color)
    track_after = pal.dim(_TRACK * max(0, TRACK_WIDTH - start - width))
    return track_before + bar + track_after


def _stage_row(
    label: str,
    duration: float,
    offset_ms: float,
    scale: float,
    is_bottleneck: bool,
    pal: Palette,
) -> str:
    track = _draw_track(offset_ms, duration, scale, _STAGE_COLOR.get(label, _DIM), pal)
    label_cell = f"  {label.ljust(LABEL_WIDTH)}"
    ms_cell = _fmt_ms(duration).rjust(8)
    row = f"{label_cell} {track}  {ms_cell}"
    if is_bottleneck:
        row += "  " + pal.bold(pal.fg("← bottleneck", 203))
    return row


def _tool_label(name: str) -> str:
    """A ``⚙ name`` label padded/truncated to LABEL_WIDTH for alignment."""
    raw = f"⚙ {name}"
    if len(raw) > LABEL_WIDTH:
        raw = raw[: LABEL_WIDTH - 1] + "…"
    return raw.ljust(LABEL_WIDTH)


def _tool_row(call: FunctionCall, turn_t0: float, scale: float, pal: Palette) -> str:
    """Render one function call as its own waterfall row (⚙ label + bar + ms)."""
    label_cell = f"  {_tool_label(call.name)}"
    offset_ms = (call.start_t - turn_t0) * 1000.0
    dur = call.duration

    if dur is None:
        # Unfinished: a single marker at the start offset, flagged on the right.
        start = min(max(round(offset_ms * scale), 0), max(0, TRACK_WIDTH - 1))
        track = (
            pal.dim(_TRACK * start)
            + pal.fg(_FILLED, _TOOL_COLOR)
            + pal.dim(_TRACK * max(0, TRACK_WIDTH - start - 1))
        )
        ms_cell = "—".rjust(8)
        return f"{label_cell} {track}  {ms_cell}  " + pal.fg("⚠ (no result)", 214)

    track = _draw_track(offset_ms, dur, scale, _TOOL_COLOR, pal)
    ms_cell = _fmt_ms(dur).rjust(8)
    return f"{label_cell} {track}  {ms_cell}"


def render_turn(
    turn: Turn, target_ms: float, global_max: float, pal: Palette
) -> str:
    scale = _scale(global_max)
    bottleneck = turn.bottleneck

    if not math.isfinite(turn.total) or turn.total < 0:
        verdict = pal.fg("⚠ total unavailable (bad timestamps)", 214)
    elif turn.total <= target_ms:
        verdict = pal.fg(f"✅ under {target_ms:.0f} ms target", 78)
    else:
        over = turn.total - target_ms
        verdict = pal.fg(
            f"❌ over {target_ms:.0f} ms target by {over:.0f} ms", 203
        )

    header = (
        f"{pal.bold(f'Turn {turn.turn_id}')}"
        f"   total {pal.bold(_fmt_ms(turn.total))}   {verdict}"
    )

    lines = [header]
    offset = 0.0
    for label in STAGE_LABELS:
        dur = turn.stages[label]
        lines.append(
            _stage_row(label, dur, offset, scale, label == bottleneck, pal)
        )
        offset += dur

    # Function/tool calls, one row each, positioned by their real timestamps.
    for call in sorted(turn.calls, key=lambda c: c.start_t):
        lines.append(_tool_row(call, turn.t0, scale, pal))

    # Ruler with a target marker.
    lines.append(_ruler(target_ms, global_max, pal))
    return "\n".join(lines)


def _ruler(target_ms: float, global_max: float, pal: Palette) -> str:
    """A dim vertical marker showing where the target line falls."""
    scale = _scale(global_max)
    pos = min(TRACK_WIDTH, round(target_ms * scale))
    pad = " " * (LABEL_WIDTH + 3)  # align with the start of the track
    marker = " " * pos + "┆"
    caption = f"target {target_ms:.0f} ms"
    # Nudge the caption so it stays under the marker without overflowing.
    caption_indent = max(0, pos - len(caption) + 1)
    caption_line = " " * caption_indent + caption
    return pad + pal.dim(marker) + "\n" + pad + pal.dim(caption_line)


def render_summary(rows: Sequence[StageStats], n_turns: int, pal: Palette) -> str:
    title = pal.bold(f"Summary  ({n_turns} turn{'s' if n_turns != 1 else ''})")
    head = f"  {'':<{LABEL_WIDTH}} {'p50':>9} {'p95':>9}"
    lines = [title, pal.dim(head)]
    for row in rows:
        if row.label == "Total":
            lines.append("  " + pal.dim("─" * (LABEL_WIDTH + 20)))
        # ljust on the raw label first; color codes would break width math.
        raw = row.label.ljust(LABEL_WIDTH)
        cell = pal.bold(raw) if row.label == "Total" else raw
        lines.append(
            f"  {cell} {_fmt_ms(row.p50):>9} {_fmt_ms(row.p95):>9}"
        )
    return "\n".join(lines)


def render_tool_summary(rows: Sequence[StageStats], pal: Palette) -> str:
    """Function-call p50/p95 block, appended only when tool calls exist."""
    width = max(LABEL_WIDTH, max(len(r.label) for r in rows))
    n_turns = rows[0].count
    n_calls = sum(r.count for r in rows[1:])
    title = pal.bold(
        f"Tools  ({n_calls} call{'s' if n_calls != 1 else ''} "
        f"in {n_turns} turn{'s' if n_turns != 1 else ''})"
    )
    head = f"  {'':<{width}} {'p50':>9} {'p95':>9}"
    lines = [title, pal.dim(head)]
    for i, row in enumerate(rows):
        raw = row.label.ljust(width)
        cell = pal.bold(raw) if i == 0 else raw
        lines.append(f"  {cell} {_fmt_ms(row.p50):>9} {_fmt_ms(row.p95):>9}")
    return "\n".join(lines)


def render_replay(
    turns: Sequence[Turn], target_ms: float, source: str, pal: Palette
) -> str:
    if not turns:
        return pal.dim("No complete turns to display.")

    global_max = max(max(t.total for t in turns), target_ms)
    blocks = [pal.bold("voxprofile") + pal.dim(f"  ·  replay  ·  {source}"), ""]
    for turn in turns:
        blocks.append(render_turn(turn, target_ms, global_max, pal))
        blocks.append("")

    rows = aggregate(turns)
    blocks.append(render_summary(rows, len(turns), pal))

    tool_rows = aggregate_tools(turns)
    if tool_rows:
        blocks.append("")
        blocks.append(render_tool_summary(tool_rows, pal))
    return "\n".join(blocks)


def render_stats(
    rows: Sequence[StageStats], sources: Sequence[str], n_turns: int, pal: Palette,
    tool_rows: Sequence[StageStats] = (),
) -> str:
    """Full p50/p95/min/max table for the ``stats`` subcommand."""
    title = pal.bold("voxprofile") + pal.dim("  ·  stats")
    src_line = pal.dim(
        f"{len(sources)} file(s), {n_turns} turn(s): "
        + ", ".join(sources)
    )
    cols = ("p50", "p95", "min", "max")
    header = f"  {'stage':<{LABEL_WIDTH}}" + "".join(c.rjust(10) for c in cols)
    lines = [title, src_line, "", pal.dim(header)]
    for row in rows:
        if row.label == "Total":
            lines.append("  " + pal.dim("─" * (LABEL_WIDTH + 40)))
        raw = row.label.ljust(LABEL_WIDTH)
        cell = pal.bold(raw) if row.label == "Total" else raw
        values = "".join(
            _fmt_ms(v).rjust(10) for v in (row.p50, row.p95, row.min, row.max)
        )
        lines.append(f"  {cell}{values}")

    if tool_rows:
        width = max(LABEL_WIDTH, max(len(r.label) for r in tool_rows))
        n_calls = sum(r.count for r in tool_rows[1:])
        lines.append("")
        lines.append(pal.bold(f"tools  ({n_calls} call(s))"))
        tool_head = f"  {'function':<{width}}" + "".join(c.rjust(10) for c in cols)
        lines.append(pal.dim(tool_head))
        for i, row in enumerate(tool_rows):
            raw = row.label.ljust(width)
            cell = pal.bold(raw) if i == 0 else raw
            values = "".join(
                _fmt_ms(v).rjust(10) for v in (row.p50, row.p95, row.min, row.max)
            )
            lines.append(f"  {cell}{values}")
    return "\n".join(lines)
