from voxprofile.html import render_html
from voxprofile.model import Turn
from voxprofile.render import (
    TRACK_WIDTH,
    Palette,
    render_replay,
    render_stats,
    render_turn,
)
from voxprofile.stats import aggregate


def _turn(tid, stt, llm, tts, pb):
    stages = {"STT": stt, "LLM": llm, "TTS": tts, "Playback": pb}
    return Turn(turn_id=tid, stages=stages, total=stt + llm + tts + pb)


PLAIN = Palette(enabled=False)


def test_render_replay_plain_has_bars_and_bottleneck():
    turns = [_turn(1, 120, 850, 100, 60)]  # LLM is the bottleneck, over target
    out = render_replay(turns, 800.0, "x.jsonl", PLAIN)
    assert "Turn 1" in out
    assert "← bottleneck" in out
    assert "█" in out
    assert "❌" in out
    assert "Summary" in out


def test_render_replay_pass_marker():
    turns = [_turn(1, 100, 300, 90, 50)]  # total 540 under target
    out = render_replay(turns, 800.0, "x.jsonl", PLAIN)
    assert "✅" in out


def test_no_color_has_no_ansi():
    turns = [_turn(1, 120, 850, 100, 60)]
    out = render_replay(turns, 800.0, "x.jsonl", PLAIN)
    assert "\033[" not in out


def test_color_palette_emits_ansi():
    pal = Palette(enabled=True)
    turns = [_turn(1, 120, 850, 100, 60)]
    out = render_replay(turns, 800.0, "x.jsonl", pal)
    assert "\033[" in out


def test_render_stats_table():
    turns = [_turn(1, 100, 300, 90, 50), _turn(2, 200, 500, 110, 60)]
    rows = aggregate(turns)
    out = render_stats(rows, ["a.jsonl", "b.jsonl"], len(turns), PLAIN)
    assert "p50" in out and "p95" in out and "min" in out and "max" in out
    assert "Total" in out


def test_html_is_self_contained():
    turns = [_turn(1, 120, 850, 100, 60), _turn(2, 100, 300, 90, 50)]
    doc = render_html(turns, 800.0, "x.jsonl")
    assert doc.startswith("<!DOCTYPE html>")
    assert "<style>" in doc
    # no external resources
    assert "http://" not in doc and "https://" not in doc
    assert "src=" not in doc and "<link" not in doc
    assert "bottleneck" in doc
    assert "<table>" in doc


def test_html_empty_turns():
    doc = render_html([], 800.0, "x.jsonl")
    assert "No complete turns" in doc


def test_out_of_order_offsets_do_not_overflow_track():
    # A negative stage duration makes cumulative offsets go negative; the
    # rendered track must still be exactly TRACK_WIDTH wide on every stage row.
    stages = {"STT": -50, "LLM": 300, "TTS": 90, "Playback": 50}
    turn = Turn(turn_id=1, stages=stages, total=390)
    out = render_turn(turn, 800.0, 800.0, PLAIN)
    for line in out.splitlines():
        track = "".join(c for c in line if c in "█░")
        if track:  # skip header/ruler lines that carry no track glyphs
            assert len(track) == TRACK_WIDTH


def test_negative_total_gets_neutral_verdict():
    stages = {"STT": 100, "LLM": -900, "TTS": 90, "Playback": 50}
    turn = Turn(turn_id=1, stages=stages, total=-660)
    out = render_turn(turn, 800.0, 800.0, PLAIN)
    assert "✅" not in out
    assert "⚠" in out
