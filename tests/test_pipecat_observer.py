"""Unit tests for the Pipecat observer using synthetic frames.

These tests do not require Pipecat, a real bot, or any API keys. They feed
lightweight frame objects (whose class name matches the real Pipecat frames)
through the observer and assert the recorded JSONL round-trips into the
expected turns.
"""

import asyncio
import json

import pytest

from voxprofile.model import load_turns
from voxprofile.pipecat_observer import HAVE_PIPECAT, VoxprofileObserver


def _frame(name):
    """A synthetic frame object whose class name matches a Pipecat frame."""
    return type(name, (), {})()


def _func_frame(name, function_name, tool_call_id):
    """A synthetic FunctionCall frame carrying function_name/tool_call_id."""
    return type(name, (), {
        "function_name": function_name,
        "tool_call_id": tool_call_id,
    })()


class _Direction:
    def __init__(self, name):
        self.name = name


class _FramePushed:
    """Mimics pipecat.observers.base_observer.FramePushed."""

    def __init__(self, frame, direction="DOWNSTREAM"):
        self.frame = frame
        self.direction = _Direction(direction)


def test_full_turn_derives_correct_stages(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    t0 = 1000.0
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=t0)
    obs.handle_frame(_frame("TranscriptionFrame"), t=t0 + 0.150)
    obs.handle_frame(_frame("LLMTextFrame"), t=t0 + 0.550)
    obs.handle_frame(_frame("LLMTextFrame"), t=t0 + 0.560)  # later tokens ignored
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=t0 + 0.660)
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=t0 + 0.700)  # ignored
    obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=t0 + 0.720)
    obs.close()

    turns = load_turns(str(path))
    assert len(turns) == 1
    stages = turns[0].stages
    assert round(stages["STT"]) == 150
    assert round(stages["LLM"]) == 400
    assert round(stages["TTS"]) == 110
    assert round(stages["Playback"]) == 60
    assert round(turns[0].total) == 720


def test_multiple_turns_increment_id(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    for i in range(2):
        base = 1000.0 + i * 10
        obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=base)
        obs.handle_frame(_frame("TranscriptionFrame"), t=base + 0.1)
        obs.handle_frame(_frame("LLMTextFrame"), t=base + 0.3)
        obs.handle_frame(_frame("TTSAudioRawFrame"), t=base + 0.4)
        obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=base + 0.45)
    obs.close()

    turns = load_turns(str(path))
    assert [t.turn_id for t in turns] == [1, 2]


def test_vad_and_userstopped_dedupe_to_one_start(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.handle_frame(_frame("UserStoppedSpeakingFrame"), t=1000.05)  # same turn
    obs.handle_frame(_frame("TranscriptionFrame"), t=1000.1)
    obs.handle_frame(_frame("LLMTextFrame"), t=1000.3)
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=1000.4)
    obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=1000.45)
    obs.close()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    starts = [r for r in lines if r["event"] == "user_stopped_speaking"]
    assert len(starts) == 1
    assert all(r["turn_id"] == 1 for r in lines)


def test_barge_in_supersedes_incomplete_turn(tmp_path):
    """A barge-in must open a fresh turn, not leak events into the old one."""
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    # Turn 1 progresses (stt + llm) then is interrupted -- no playback_started.
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.handle_frame(_frame("TranscriptionFrame"), t=1000.1)
    obs.handle_frame(_frame("LLMTextFrame"), t=1000.3)
    # User barges in ~5s later; this is a brand new, complete turn.
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1005.0)
    obs.handle_frame(_frame("TranscriptionFrame"), t=1005.1)
    obs.handle_frame(_frame("LLMTextFrame"), t=1005.3)
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=1005.4)
    obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=1005.45)
    obs.close()

    turns = load_turns(str(path))  # incomplete turn 1 is skipped
    assert len(turns) == 1
    assert turns[0].turn_id == 2
    assert round(turns[0].total) == 450
    # No absurd cross-turn duration leaked in (the old bug produced ~5000ms).
    assert turns[0].stages["TTS"] < 200

    # The recording itself must contain a distinct turn 2 with its own start.
    ids = {json.loads(l)["turn_id"] for l in path.read_text().splitlines()}
    assert ids == {1, 2}


def test_unmapped_frames_and_pre_turn_events_ignored(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    # events before any start frame are ignored
    obs.handle_frame(_frame("TranscriptionFrame"), t=999.0)
    obs.handle_frame(_frame("SomeUnrelatedFrame"), t=999.5)
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.handle_frame(_frame("SomeUnrelatedFrame"), t=1000.1)  # ignored
    obs.handle_frame(_frame("TranscriptionFrame"), t=1000.2)
    obs.close()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    events = [r["event"] for r in lines]
    assert events == ["user_stopped_speaking", "stt_final"]


def test_on_push_frame_async_and_direction_filter(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))

    async def drive():
        # upstream frames are dropped
        await obs.on_push_frame(
            _FramePushed(_frame("VADUserStoppedSpeakingFrame"), direction="UPSTREAM")
        )
        # downstream full turn
        for name in (
            "VADUserStoppedSpeakingFrame",
            "TranscriptionFrame",
            "LLMTextFrame",
            "TTSAudioRawFrame",
            "BotStartedSpeakingFrame",
        ):
            await obs.on_push_frame(_FramePushed(_frame(name)))

    asyncio.run(drive())
    obs.close()

    turns = load_turns(str(path))
    assert len(turns) == 1
    assert turns[0].turn_id == 1


def test_function_call_frames_recorded_and_paired(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.handle_frame(_frame("TranscriptionFrame"), t=1000.15)
    obs.handle_frame(
        _func_frame("FunctionCallInProgressFrame", "get_weather", "c1"), t=1000.2
    )
    obs.handle_frame(
        _func_frame("FunctionCallResultFrame", "get_weather", "c1"), t=1000.82
    )
    obs.handle_frame(_frame("LLMTextFrame"), t=1000.9)
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=1001.0)
    obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=1001.05)
    obs.close()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    fc = [r for r in lines if r["event"].startswith("function_call")]
    assert [r["event"] for r in fc] == [
        "function_call_start",
        "function_call_result",
    ]
    assert all(r["name"] == "get_weather" and r["call_id"] == "c1" for r in fc)

    turns = load_turns(str(path))
    assert len(turns) == 1
    assert len(turns[0].calls) == 1
    assert round(turns[0].calls[0].duration) == 620


def test_function_call_outside_turn_is_ignored(tmp_path):
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    # a stray tool frame before any turn opens must not be recorded
    obs.handle_frame(_func_frame("FunctionCallInProgressFrame", "f", "c1"), t=999.0)
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.close()
    events = [json.loads(l)["event"] for l in path.read_text().splitlines()]
    assert events == ["user_stopped_speaking"]


def test_function_call_does_not_count_as_barge_in_progress(tmp_path):
    """A tool call must not itself close/reset a turn or leak across turns."""
    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))
    obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    obs.handle_frame(_frame("TranscriptionFrame"), t=1000.15)
    obs.handle_frame(
        _func_frame("FunctionCallInProgressFrame", "f", "c1"), t=1000.2
    )
    obs.handle_frame(
        _func_frame("FunctionCallResultFrame", "f", "c1"), t=1000.4
    )
    obs.handle_frame(_frame("LLMTextFrame"), t=1000.9)
    obs.handle_frame(_frame("TTSAudioRawFrame"), t=1001.0)
    obs.handle_frame(_frame("BotStartedSpeakingFrame"), t=1001.05)
    obs.close()
    turns = load_turns(str(path))
    assert len(turns) == 1
    assert turns[0].turn_id == 1
    assert len(turns[0].calls) == 1


def test_context_manager_closes(tmp_path):
    path = tmp_path / "rec.jsonl"
    with VoxprofileObserver(str(path)) as obs:
        obs.handle_frame(_frame("VADUserStoppedSpeakingFrame"), t=1000.0)
    assert path.exists()


@pytest.mark.skipif(not HAVE_PIPECAT, reason="pipecat-ai not installed")
def test_real_pipecat_frames_end_to_end(tmp_path):
    """Drive the observer with genuine Pipecat 1.5.x frame objects."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        LLMTextFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    assert issubclass(VoxprofileObserver, BaseObserver)

    path = tmp_path / "rec.jsonl"
    obs = VoxprofileObserver(str(path))

    def pushed(frame):
        return FramePushed(
            source=None,
            destination=None,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )

    frames = [
        VADUserStoppedSpeakingFrame(stop_secs=0.8, timestamp=0.0),
        TranscriptionFrame(text="hi", user_id="u1", timestamp="2026-07-05T00:00:00Z"),
        FunctionCallInProgressFrame(
            function_name="get_weather", tool_call_id="c1", arguments={"city": "SF"}
        ),
        FunctionCallResultFrame(
            function_name="get_weather", tool_call_id="c1",
            arguments={"city": "SF"}, result={"temp": 20}, run_llm=True,
        ),
        LLMTextFrame(text="Hi"),
        LLMTextFrame(text=" there"),  # later token ignored
        TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1),
        TTSAudioRawFrame(audio=b"\x01\x01", sample_rate=16000, num_channels=1),  # ignored
        BotStartedSpeakingFrame(),
    ]

    async def drive():
        for f in frames:
            await obs.on_push_frame(pushed(f))

    asyncio.run(drive())
    obs.close()

    records = [json.loads(l) for l in path.read_text().splitlines()]
    events = [r["event"] for r in records]
    assert events == [
        "user_stopped_speaking",
        "stt_final",
        "function_call_start",
        "function_call_result",
        "llm_first_token",
        "tts_first_byte",
        "playback_started",
    ]
    fc = [r for r in records if r["event"].startswith("function_call")]
    assert all(r["name"] == "get_weather" and r["call_id"] == "c1" for r in fc)

    turns = load_turns(str(path))
    assert len(turns) == 1
    assert turns[0].turn_id == 1
    assert len(turns[0].calls) == 1
    assert turns[0].calls[0].name == "get_weather"
