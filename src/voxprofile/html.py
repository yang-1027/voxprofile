"""Self-contained HTML waterfall export (inline CSS, no external assets)."""

from __future__ import annotations

import html
from typing import Sequence

from .model import STAGE_LABELS, Turn
from .stats import StageStats, aggregate

_STAGE_COLOR = {
    "STT": "#2bb6c4",
    "LLM": "#f0a437",
    "TTS": "#57b96a",
    "Playback": "#9d7be0",
}

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  background: #0f1117; color: #e6e8ef;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 880px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: .3px; }
.sub { color: #8b90a0; margin: 0 0 28px; font-size: 13px; }
.turn { background: #171a23; border: 1px solid #232735; border-radius: 10px;
  padding: 16px 18px; margin-bottom: 14px; }
.turn-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }
.turn-id { font-weight: 600; }
.total { color: #c7cbd8; }
.badge { margin-left: auto; font-size: 12px; padding: 3px 10px; border-radius: 999px; }
.pass { background: rgba(87,185,106,.15); color: #7fd695; }
.fail { background: rgba(224,92,92,.15); color: #f08a8a; }
.bar { display: flex; width: 100%; height: 26px; border-radius: 5px;
  overflow: hidden; background: #10131b; position: relative; }
.seg { position: relative; min-width: 2px; transition: filter .12s; }
.seg:hover { filter: brightness(1.25); }
.seg .tip { position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%);
  background: #000; color: #fff; padding: 3px 8px; border-radius: 5px; font-size: 12px;
  white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity .1s; z-index: 5; }
.seg:hover .tip { opacity: 1; }
.legend { display: flex; gap: 16px; margin: 6px 0 30px; color: #9aa0b2; font-size: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.target { position: absolute; top: -4px; bottom: -4px; width: 2px;
  background: repeating-linear-gradient(#e6e8ef 0 4px, transparent 4px 8px); }
.bn { font-size: 11px; color: #f08a8a; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 8px 12px; border-bottom: 1px solid #232735; }
th:first-child, td:first-child { text-align: left; }
thead th { color: #8b90a0; font-weight: 500; font-size: 12px; }
tr.total-row td { font-weight: 600; border-top: 2px solid #313648; }
h2 { font-size: 15px; margin: 32px 0 4px; }
"""


def _esc(s: str) -> str:
    return html.escape(str(s))


def _fmt(ms: float) -> str:
    return f"{ms:.0f} ms"


def _turn_html(turn: Turn, target_ms: float, global_max: float) -> str:
    passed = turn.total <= target_ms
    badge_cls = "pass" if passed else "fail"
    badge_txt = (
        f"under {target_ms:.0f} ms" if passed
        else f"over by {turn.total - target_ms:.0f} ms"
    )
    bottleneck = turn.bottleneck

    segs = []
    for label in STAGE_LABELS:
        dur = turn.stages[label]
        pct = (dur / turn.total * 100.0) if turn.total > 0 else 0.0
        color = _STAGE_COLOR.get(label, "#888")
        tip = f"{label} · {_fmt(dur)}"
        segs.append(
            f'<div class="seg" style="width:{pct:.3f}%;background:{color}">'
            f'<div class="tip">{_esc(tip)}</div></div>'
        )

    # Target marker positioned relative to the widest turn (or the target).
    target_pct = min(100.0, target_ms / turn.total * 100.0) if turn.total > 0 else 0.0
    target_marker = (
        f'<div class="target" style="left:{target_pct:.3f}%"></div>'
        if target_pct < 100.0 else ""
    )

    bn_line = (
        f'<div class="bn">← bottleneck: {_esc(bottleneck)} '
        f'({_fmt(turn.stages[bottleneck])})</div>'
    )

    return (
        '<div class="turn">'
        '<div class="turn-head">'
        f'<span class="turn-id">Turn {_esc(turn.turn_id)}</span>'
        f'<span class="total">total {_fmt(turn.total)}</span>'
        f'<span class="badge {badge_cls}">{_esc(badge_txt)}</span>'
        "</div>"
        f'<div class="bar">{"".join(segs)}{target_marker}</div>'
        f"{bn_line}"
        "</div>"
    )


def _legend_html() -> str:
    items = []
    for label in STAGE_LABELS:
        color = _STAGE_COLOR.get(label, "#888")
        items.append(
            f'<span><span class="dot" style="background:{color}"></span>{_esc(label)}</span>'
        )
    items.append('<span><span class="dot" style="background:#e6e8ef"></span>target line</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def _summary_html(rows: Sequence[StageStats]) -> str:
    head = (
        "<thead><tr><th>Stage</th><th>p50</th><th>p95</th>"
        "<th>min</th><th>max</th></tr></thead>"
    )
    body = []
    for row in rows:
        cls = ' class="total-row"' if row.label == "Total" else ""
        body.append(
            f"<tr{cls}><td>{_esc(row.label)}</td>"
            f"<td>{_fmt(row.p50)}</td><td>{_fmt(row.p95)}</td>"
            f"<td>{_fmt(row.min)}</td><td>{_fmt(row.max)}</td></tr>"
        )
    return f"<table>{head}<tbody>{''.join(body)}</tbody></table>"


def render_html(turns: Sequence[Turn], target_ms: float, source: str) -> str:
    """Return a complete, self-contained HTML document as a string."""
    if turns:
        global_max = max(max(t.total for t in turns), target_ms)
        turn_blocks = "".join(
            _turn_html(t, target_ms, global_max) for t in turns
        )
        summary = _summary_html(aggregate(turns))
        n = len(turns)
    else:
        turn_blocks = '<p class="sub">No complete turns to display.</p>'
        summary = ""
        n = 0

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>voxprofile latency waterfall</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        "<h1>voxprofile · latency waterfall</h1>"
        f'<p class="sub">{_esc(source)} — {n} turn(s), target {target_ms:.0f} ms</p>'
        f"{_legend_html()}"
        f"{turn_blocks}"
        f"<h2>Summary</h2>{summary}"
        "</div></body></html>\n"
    )
