import json
import os

import pytest

from voxprofile.cli import main

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples"
)
SAMPLE = os.path.join(EXAMPLES, "sample_events.jsonl")
SAMPLE2 = os.path.join(EXAMPLES, "sample_events_run2.jsonl")


def test_replay_runs(capsys):
    rc = main(["replay", SAMPLE, "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Turn 1" in out
    assert "Summary" in out
    assert "← bottleneck" in out


def test_replay_target_flag_changes_verdict(capsys):
    main(["replay", SAMPLE, "--no-color", "--target", "2000"])
    out = capsys.readouterr().out
    assert "❌" not in out  # everything passes under a huge target


def test_replay_html_export(tmp_path, capsys):
    html_path = tmp_path / "out.html"
    rc = main(["replay", SAMPLE, "--no-color", "--html", str(html_path)])
    assert rc == 0
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "https://" not in text


def test_stats_multi_file(capsys):
    rc = main(["stats", SAMPLE, SAMPLE2, "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "p50" in out and "p95" in out
    assert "Total" in out
    assert "2 file(s)" in out


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        main([])


def test_replay_missing_file_returns_2(tmp_path, capsys):
    rc = main(["replay", str(tmp_path / "nope.jsonl"), "--no-color"])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_replay_directory_is_handled(tmp_path, capsys):
    rc = main(["replay", str(tmp_path), "--no-color"])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_stats_skips_bad_file_keeps_going(tmp_path, capsys):
    rc = main(["stats", SAMPLE, str(tmp_path / "nope.jsonl"), "--no-color"])
    out, err = capsys.readouterr()
    assert "cannot read" in err
    assert "Total" in out  # still produced output from the good file
    assert rc == 2


def test_replay_html_skipped_when_no_complete_turns(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"turn_id": 1, "event": "user_stopped_speaking", "t": 1.0}) + "\n",
        encoding="utf-8",
    )
    html_path = tmp_path / "out.html"
    rc = main(["replay", str(bad), "--no-color", "--html", str(html_path)])
    err = capsys.readouterr().err
    assert not html_path.exists()
    assert "skipping HTML" in err
    assert rc == 1
