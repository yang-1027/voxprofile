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
