import json

from voxprofile.model import STAGE_LABELS, load_turns, load_turns_multi


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def _full_turn(turn_id, t0, stt, llm, tts, pb):
    cum = 0.0
    out = [{"turn_id": turn_id, "event": "user_stopped_speaking", "t": t0}]
    for name, d in [
        ("stt_final", stt),
        ("llm_first_token", llm),
        ("tts_first_byte", tts),
        ("playback_started", pb),
    ]:
        cum += d
        out.append({"turn_id": turn_id, "event": name, "t": t0 + cum})
    return out


def test_stage_durations_in_ms(tmp_path):
    p = _write(tmp_path / "e.jsonl", _full_turn(1, 1000.0, 0.12, 0.30, 0.09, 0.05))
    turns = load_turns(p)
    assert len(turns) == 1
    t = turns[0]
    assert round(t.stages["STT"]) == 120
    assert round(t.stages["LLM"]) == 300
    assert round(t.stages["TTS"]) == 90
    assert round(t.stages["Playback"]) == 50
    assert round(t.total) == 560
    assert t.bottleneck == "LLM"


def test_out_of_order_lines_are_grouped(tmp_path):
    recs = _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04)
    recs = list(reversed(recs))  # shuffle order within file
    p = _write(tmp_path / "e.jsonl", recs)
    turns = load_turns(p)
    assert len(turns) == 1
    assert round(turns[0].stages["LLM"]) == 200


def test_missing_event_skips_turn(tmp_path, capsys):
    recs = _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04)
    recs += _full_turn(2, 2000.0, 0.1, 0.2, 0.08, 0.04)[:-1]  # drop last event
    p = _write(tmp_path / "e.jsonl", recs)
    turns = load_turns(p)
    assert [t.turn_id for t in turns] == [1]
    err = capsys.readouterr().err
    assert "skipping turn 2" in err
    assert "playback_started" in err


def test_malformed_json_does_not_crash(tmp_path, capsys):
    path = tmp_path / "e.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json\n")
        for r in _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04):
            fh.write(json.dumps(r) + "\n")
    turns = load_turns(str(path))
    assert len(turns) == 1
    assert "malformed JSON" in capsys.readouterr().err


def test_load_multi_concatenates(tmp_path):
    p1 = _write(tmp_path / "a.jsonl", _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04))
    p2 = _write(tmp_path / "b.jsonl", _full_turn(1, 2000.0, 0.1, 0.2, 0.08, 0.04))
    turns = load_turns_multi([p1, p2])
    assert len(turns) == 2
    assert {t.source for t in turns} == {p1, p2}


def test_stage_labels_constant():
    assert STAGE_LABELS == ["STT", "LLM", "TTS", "Playback"]


def test_unhashable_turn_id_does_not_crash(tmp_path, capsys):
    path = tmp_path / "e.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        # a list/object turn_id must not blow up parsing
        fh.write(
            json.dumps({"turn_id": [1, 2], "event": "user_stopped_speaking", "t": 1.0})
            + "\n"
        )
        fh.write(
            json.dumps({"turn_id": {"x": 1}, "event": "stt_final", "t": 1.1}) + "\n"
        )
        for r in _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04):
            fh.write(json.dumps(r) + "\n")
    turns = load_turns(str(path))  # must never raise
    assert len(turns) == 1
    assert "non-scalar turn_id" in capsys.readouterr().err


def test_int_and_str_turn_id_merge(tmp_path):
    recs = [
        {"turn_id": 1, "event": "user_stopped_speaking", "t": 1000.0},
        {"turn_id": "1", "event": "stt_final", "t": 1000.1},
        {"turn_id": 1, "event": "llm_first_token", "t": 1000.3},
        {"turn_id": "1", "event": "tts_first_byte", "t": 1000.4},
        {"turn_id": 1, "event": "playback_started", "t": 1000.45},
    ]
    p = _write(tmp_path / "e.jsonl", recs)
    turns = load_turns(p)
    assert len(turns) == 1  # not split across int 1 and str "1"
    assert round(turns[0].total) == 450


def test_duplicate_event_warns(tmp_path, capsys):
    recs = _full_turn(1, 1000.0, 0.1, 0.2, 0.08, 0.04)
    recs.append({"turn_id": 1, "event": "stt_final", "t": 1000.5})  # dup, wins
    p = _write(tmp_path / "e.jsonl", recs)
    turns = load_turns(p)
    assert len(turns) == 1
    assert "duplicate" in capsys.readouterr().err
    # latest value kept: stt_final now at +0.5s
    assert round(turns[0].stages["STT"]) == 500
