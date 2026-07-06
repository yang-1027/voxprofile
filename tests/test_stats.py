import math

from voxprofile.model import Turn
from voxprofile.stats import aggregate, percentile


def test_percentile_matches_linear_method():
    data = [1, 2, 3, 4]
    assert percentile(data, 50) == 2.5
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 4
    # p95 with linear interpolation
    assert math.isclose(percentile(data, 95), 3.85)


def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 50))


def test_percentile_single_value():
    assert percentile([42], 95) == 42


def _turn(tid, stt, llm, tts, pb):
    stages = {"STT": stt, "LLM": llm, "TTS": tts, "Playback": pb}
    return Turn(turn_id=tid, stages=stages, total=stt + llm + tts + pb)


def test_aggregate_rows_and_total():
    turns = [
        _turn(1, 100, 300, 90, 50),
        _turn(2, 200, 500, 110, 60),
    ]
    rows = aggregate(turns)
    labels = [r.label for r in rows]
    assert labels == ["STT", "LLM", "TTS", "Playback", "Total"]
    total_row = rows[-1]
    assert total_row.min == 540
    assert total_row.max == 870
    assert total_row.count == 2
    stt_row = rows[0]
    assert stt_row.p50 == 150  # midpoint of 100 and 200
