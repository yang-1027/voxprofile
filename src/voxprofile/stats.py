"""Percentile aggregation over turns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .model import STAGE_LABELS, Turn


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method).

    ``p`` is in the range [0, 100]. Returns NaN for an empty input.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


@dataclass
class StageStats:
    label: str
    p50: float
    p95: float
    min: float
    max: float
    count: int


def _summarize(label: str, values: Sequence[float]) -> StageStats:
    if not values:
        nan = float("nan")
        return StageStats(label, nan, nan, nan, nan, 0)
    return StageStats(
        label=label,
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        min=min(values),
        max=max(values),
        count=len(values),
    )


def aggregate(turns: Sequence[Turn]) -> list[StageStats]:
    """Return per-stage stats followed by a final ``Total`` row."""
    rows: list[StageStats] = []
    for label in STAGE_LABELS:
        rows.append(_summarize(label, [t.stages[label] for t in turns]))
    rows.append(_summarize("Total", [t.total for t in turns]))
    return rows


# Label used for the aggregate "sum of tool time per turn" row.
TOOL_TOTAL_LABEL = "total/turn"


def aggregate_tools(turns: Sequence[Turn]) -> list[StageStats]:
    """Function-call latency stats, additive to :func:`aggregate`.

    Returns an empty list when no turn has a *finished* function call, so the
    tool section is only ever shown when there is something to show. Otherwise
    the first row is ``total/turn`` (per-turn summed tool time over tool-using
    turns) followed by one row per distinct function name, in first-seen order.
    Unfinished calls carry no duration and are excluded from the statistics.
    """
    per_turn_totals: list[float] = []
    by_name: dict[str, list[float]] = {}
    for turn in turns:
        turn_total = 0.0
        used_tools = False
        for call in turn.calls:
            dur = call.duration
            if dur is None:
                continue
            used_tools = True
            turn_total += dur
            by_name.setdefault(call.name, []).append(dur)
        if used_tools:
            per_turn_totals.append(turn_total)

    if not per_turn_totals:
        return []

    rows = [_summarize(TOOL_TOTAL_LABEL, per_turn_totals)]
    for name, durations in by_name.items():
        rows.append(_summarize(name, durations))
    return rows
