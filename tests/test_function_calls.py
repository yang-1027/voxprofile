"""Regression tests for function/tool-call capture and visualization.

These cover the additive function-call feature end to end: model pairing,
waterfall + HTML rendering, cross-run stats, and -- crucially -- that inputs
with zero function calls behave exactly as before.
"""

import json
import os
import re

from voxprofile.html import render_html
from voxprofile.model import FunctionCall, load_turns
from voxprofile.render import Palette, render_replay, render_turn
from voxprofile.stats import aggregate, aggregate_tools

PLAIN = Palette(enabled=False)

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
SAMPLE = os.path.join(EXAMPLES, "sample_events.jsonl")
SAMPLE_TOOLS = os.path.join(EXAMPLES, "sample_events_tools.jsonl")


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def _turn_events(turn_id, t0, calls=()):
    """A full 5-event turn, optionally interleaved with function-call events."""
    recs = [
        {"turn_id": turn_id, "event": "user_stopped_speaking", "t": t0},
        {"turn_id": turn_id, "event": "stt_final", "t": t0 + 0.15},
    ]
    recs.extend(calls)
    recs.extend(
        [
            {"turn_id": turn_id, "event": "llm_first_token", "t": t0 + 0.9},
            {"turn_id": turn_id, "event": "tts_first_byte", "t": t0 + 1.0},
            {"turn_id": turn_id, "event": "playback_started", "t": t0 + 1.05},
        ]
    )
    return recs


# --- model: pairing & duration -----------------------------------------------

def test_single_call_paired_by_id(tmp_path):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.2,
         "name": "get_weather", "call_id": "c1"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.82,
         "name": "get_weather", "call_id": "c1"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    assert len(turns) == 1
    assert len(turns[0].calls) == 1
    call = turns[0].calls[0]
    assert call.name == "get_weather"
    assert call.finished
    assert round(call.duration) == 620
    # boundary stages are untouched by the interleaved call
    assert round(turns[0].stages["STT"]) == 150


def test_chained_calls_multiple_rows(tmp_path):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.18,
         "name": "search_docs", "call_id": "a"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.56,
         "name": "search_docs", "call_id": "a"},
        {"turn_id": 1, "event": "function_call_start", "t": 1000.58,
         "name": "rank_results", "call_id": "b"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.80,
         "name": "rank_results", "call_id": "b"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    got = [(c.name, round(c.duration)) for c in turns[0].calls]
    assert got == [("search_docs", 380), ("rank_results", 220)]


def test_start_without_result_is_unfinished(tmp_path):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.2,
         "name": "slow_query", "call_id": "z"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    assert len(turns[0].calls) == 1
    call = turns[0].calls[0]
    assert not call.finished
    assert call.duration is None


def test_fifo_pairing_without_call_id(tmp_path):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.20, "name": "f"},
        {"turn_id": 1, "event": "function_call_start", "t": 1000.30, "name": "g"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.50, "name": "f"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.70, "name": "g"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    got = [(c.name, round(c.duration)) for c in turns[0].calls]
    # earliest start pairs with earliest result, etc.
    assert got == [("f", 300), ("g", 400)]


def test_start_id_result_no_id_still_pairs(tmp_path, capsys):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.2,
         "name": "f", "call_id": "c1"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.6, "name": "f"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    err = capsys.readouterr().err
    assert len(turns[0].calls) == 1
    call = turns[0].calls[0]
    assert call.finished and round(call.duration) == 400
    assert "without a matching start" not in err


def test_result_id_start_no_id_still_pairs(tmp_path):
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.2, "name": "f"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.6,
         "name": "f", "call_id": "c1"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    assert len(turns[0].calls) == 1
    call = turns[0].calls[0]
    assert call.finished and round(call.duration) == 400


def test_backwards_pair_by_id_warns_excluded_and_marked(tmp_path, capsys):
    # id matches but the result timestamp precedes the start (clock skew).
    calls = [
        {"turn_id": 1, "event": "function_call_start", "t": 1000.5,
         "name": "skewed", "call_id": "c1"},
        {"turn_id": 1, "event": "function_call_result", "t": 1000.2,
         "name": "skewed", "call_id": "c1"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    err = capsys.readouterr().err
    assert "out-of-order" in err
    call = turns[0].calls[0]
    assert call.duration < 0  # the raw (negative) value is retained on the model
    # excluded from stats
    assert aggregate_tools(turns) == []
    # rendered as a neutral marker, never a "-Nms" bar
    out = render_turn(turns[0], 800.0, 1200.0, PLAIN)
    assert "bad timing" in out
    assert not re.search(r"-\d+\s*ms", out)
    # HTML likewise avoids a negative segment
    doc = render_html(turns, 800.0, "x.jsonl")
    assert "bad timing" in doc
    assert not re.search(r"-\d+\s*ms", doc)


def test_fifo_backwards_pair_warns_not_orphan(tmp_path, capsys):
    # a stray earlier result FIFO-pairs with a later start -> negative duration;
    # this used to slip in silently with no warning at all.
    calls = [
        {"turn_id": 1, "event": "function_call_result", "t": 1000.10, "name": "s"},
        {"turn_id": 1, "event": "function_call_start", "t": 1000.30, "name": "s"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    err = capsys.readouterr().err
    assert "out-of-order" in err
    assert "without a matching start" not in err  # they did pair (FIFO)
    assert aggregate_tools(turns) == []  # negative duration excluded from stats


def test_orphan_result_is_dropped_not_crash(tmp_path, capsys):
    calls = [
        {"turn_id": 1, "event": "function_call_result", "t": 1000.5,
         "name": "ghost", "call_id": "x"},
    ]
    p = _write(tmp_path / "e.jsonl", _turn_events(1, 1000.0, calls))
    turns = load_turns(p)
    assert turns[0].calls == []
    assert "without a matching start" in capsys.readouterr().err


def test_malformed_call_line_skipped(tmp_path, capsys):
    path = tmp_path / "e.jsonl"
    recs = _turn_events(1, 1000.0)
    # inject a broken function-call record (non-numeric t) plus a good one
    recs.insert(2, {"turn_id": 1, "event": "function_call_start", "t": "oops",
                    "name": "bad"})
    recs.insert(3, {"turn_id": 1, "event": "function_call_start", "t": 1000.2,
                    "name": "", "call_id": "c1"})  # missing name -> "?"
    recs.insert(4, {"turn_id": 1, "event": "function_call_result", "t": 1000.4,
                    "name": "ok", "call_id": "c1"})
    _write(path, recs)
    turns = load_turns(str(path))  # must not raise
    assert len(turns) == 1
    names = [c.name for c in turns[0].calls]
    assert names == ["?"]  # the bad-t record was skipped
    assert "non-numeric t" in capsys.readouterr().err


# --- backward compatibility --------------------------------------------------

def test_existing_sample_has_no_calls():
    turns = load_turns(SAMPLE)
    assert turns
    assert all(t.calls == [] for t in turns)
    assert aggregate_tools(turns) == []


def test_zero_call_replay_has_no_tool_artifacts():
    turns = load_turns(SAMPLE)
    out = render_replay(turns, 800.0, SAMPLE, PLAIN)
    assert "⚙" not in out
    assert "Tools" not in out


def test_zero_call_turn_render_unchanged():
    from voxprofile.model import Turn

    stages = {"STT": 120, "LLM": 300, "TTS": 90, "Playback": 50}
    no_calls = Turn(turn_id=1, stages=stages, total=560)
    out = render_turn(no_calls, 800.0, 800.0, PLAIN)
    assert "⚙" not in out
    # exactly four stage bar rows, nothing extra before the ruler
    bar_rows = [ln for ln in out.splitlines() if "█" in ln or "░" in ln]
    assert len(bar_rows) == 4


# --- render: tool rows -------------------------------------------------------

def _turn_with_calls():
    turns = load_turns(SAMPLE_TOOLS)
    return {t.turn_id: t for t in turns}


def test_tool_row_shows_name_and_duration():
    turn = _turn_with_calls()[2]
    out = render_turn(turn, 800.0, 1200.0, PLAIN)
    assert "⚙" in out
    assert "get_wea" in out  # truncated label
    assert "620 ms" in out


def test_unfinished_call_flagged_in_render():
    turn = _turn_with_calls()[5]
    out = render_turn(turn, 800.0, 1200.0, PLAIN)
    assert "no result" in out


def test_tool_rows_never_overflow_track():
    from voxprofile.render import TRACK_WIDTH

    for turn in load_turns(SAMPLE_TOOLS):
        out = render_turn(turn, 800.0, 1200.0, PLAIN)
        for line in out.splitlines():
            track = "".join(c for c in line if c in "█░")
            if track:
                assert len(track) == TRACK_WIDTH


# --- stats -------------------------------------------------------------------

def test_aggregate_tools_total_and_per_name():
    turns = load_turns(SAMPLE_TOOLS)
    rows = aggregate_tools(turns)
    labels = [r.label for r in rows]
    assert labels[0] == "total/turn"
    assert "get_weather" in labels
    assert "rank_results" in labels
    # per-turn total row aggregates turns 2 (620) and 3 (600); unfinished turn 5
    # contributes nothing.
    total_row = rows[0]
    assert total_row.count == 2
    assert round(total_row.min) == 600  # search_docs 380 + rank_results 220


def test_aggregate_stages_unaffected_by_tools():
    turns = load_turns(SAMPLE_TOOLS)
    rows = aggregate(turns)
    assert [r.label for r in rows] == ["STT", "LLM", "TTS", "Playback", "Total"]


# --- html --------------------------------------------------------------------

def test_html_renders_tool_segments():
    turns = load_turns(SAMPLE_TOOLS)
    doc = render_html(turns, 800.0, SAMPLE_TOOLS)
    assert "get_weather" in doc
    assert 'class="tseg"' in doc
    assert "pending" in doc  # the unfinished slow_db_query
    assert "<h2>Tools</h2>" in doc
    # user-supplied names are escaped through _esc like everything else
    assert "https://" not in doc and "src=" not in doc


def test_html_escapes_function_name(tmp_path):
    from voxprofile.model import Turn

    call = FunctionCall(name="<script>x</script>", start_t=0.2, result_t=0.5)
    turn = Turn(
        turn_id=1,
        stages={"STT": 150, "LLM": 300, "TTS": 100, "Playback": 50},
        total=1000,
        calls=[call],
        t0=0.0,
    )
    doc = render_html([turn], 800.0, "x.jsonl")
    assert "<script>x</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_html_zero_calls_has_no_tool_css():
    turns = load_turns(SAMPLE)
    doc = render_html(turns, 800.0, SAMPLE)
    assert ".tseg" not in doc
    assert "<h2>Tools</h2>" not in doc
