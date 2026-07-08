"""Pipecat integration: an observer that records voxprofile events to JSONL.

This is an *optional* integration (``pip install voxprofile[pipecat]``). The
core voxprofile CLI never imports this module, and importing it without
Pipecat installed does not raise -- :class:`VoxprofileObserver` still works for
offline unit testing and only requires Pipecat at pipeline runtime.

Event mapping (Pipecat 1.5.x frame classes -> voxprofile events)
-----------------------------------------------------------------
=========================  =========================  =========================
voxprofile event           Pipecat frame              notes
=========================  =========================  =========================
``user_stopped_speaking``  ``VADUserStoppedSpeaking``  starts a new turn; also
                           ``Frame`` (or, as a         accepts the plain
                           fallback, ``UserStopped     ``UserStoppedSpeakingFrame``
                           SpeakingFrame``)            for pipelines without VAD
``stt_final``              ``TranscriptionFrame``      the final STT result
                                                       (``InterimTranscription
                                                       Frame`` is ignored)
``llm_first_token``        first ``LLMTextFrame``      first streamed LLM token
                           of the turn
``tts_first_byte``         first ``TTSAudioRawFrame``  first synthesized audio
                           of the turn                 chunk (TTS TTFB)
``playback_started``       ``BotStartedSpeaking        bot audio playback began;
                           Frame``                     closes the turn
=========================  =========================  =========================

In addition to the five boundary events above, two *optional* function-call
events are emitted when the pipeline uses tools:

    * ``FunctionCallInProgressFrame`` -> ``function_call_start`` (carries
      ``name`` = ``function_name`` and ``call_id`` = ``tool_call_id``)
    * ``FunctionCallResultFrame`` -> ``function_call_result`` (same fields)

They are additive: they do not open, close, or de-duplicate a turn, and they
are only recorded while a turn is open. ``voxprofile replay`` pairs them by
``call_id`` and draws each call as its own waterfall row.

Timestamps use ``time.time()`` at the moment each frame is observed, so every
event shares one consistent wall-clock (epoch seconds), which is what the
voxprofile stage math expects.

What is NOT captured reliably (documented, intentionally not mapped):
    * Exact VAD endpoint refinement. ``VADUserStoppedSpeakingFrame`` carries
      ``timestamp``/``stop_secs`` that could shift the STT start earlier; we
      use observation time instead to keep a single clock source.
    * Provider-reported ``TTFBMetricsData`` inside ``MetricsFrame``. Those give
      a *duration*, not the first-token wall-clock instant, so we derive
      ``llm_first_token``/``tts_first_byte`` from the first output frame
      instead. Enable pipeline metrics if you also want provider TTFB numbers.
    * Barge-in / interruptions. A turn interrupted before
      ``BotStartedSpeakingFrame`` is left open and simply superseded by the
      next ``user_stopped_speaking``; it will be reported as an incomplete turn
      by ``voxprofile replay`` and skipped.
"""

from __future__ import annotations

import json
import time
from typing import Optional, TextIO

# Graceful, lazy Pipecat import. When Pipecat is absent, we fall back to a
# plain ``object`` base so the module still imports and the frame-dispatch
# logic can be unit-tested with synthetic frames.
try:  # pragma: no cover - trivial import guard
    from pipecat.observers.base_observer import BaseObserver as _BaseObserver

    HAVE_PIPECAT = True
except Exception:  # pragma: no cover - exercised only without pipecat installed
    _BaseObserver = object  # type: ignore[assignment,misc]
    HAVE_PIPECAT = False


# Frame class name -> voxprofile event. Dispatching by ``type(frame).__name__``
# keeps this module free of hard imports on Pipecat's frame classes and makes
# it testable with lightweight synthetic frames.
_FRAME_EVENT = {
    "VADUserStoppedSpeakingFrame": "user_stopped_speaking",
    "UserStoppedSpeakingFrame": "user_stopped_speaking",
    "TranscriptionFrame": "stt_final",
    "LLMTextFrame": "llm_first_token",
    "TTSAudioRawFrame": "tts_first_byte",
    "BotStartedSpeakingFrame": "playback_started",
}

# Function/tool-call frames are additive: they carry ``function_name`` /
# ``tool_call_id`` and may occur 0..N times per turn. They never open, close,
# or de-duplicate a turn, so barge-in and boundary semantics are untouched.
_FUNC_FRAME_EVENT = {
    "FunctionCallInProgressFrame": "function_call_start",
    "FunctionCallResultFrame": "function_call_result",
}

_START_EVENT = "user_stopped_speaking"
_END_EVENT = "playback_started"


class VoxprofileObserver(_BaseObserver):
    """A Pipecat observer that appends voxprofile events to a JSONL file.

    Attach it to a pipeline task::

        from voxprofile.pipecat_observer import VoxprofileObserver

        observer = VoxprofileObserver("latency.jsonl")
        task = PipelineTask(pipeline, observers=[observer])
        # ... run the pipeline ...
        observer.close()

    Afterwards inspect the recording with ``voxprofile replay latency.jsonl``.

    The observer is safe to construct without a running pipeline (used by
    tests): call :meth:`handle_frame` directly with synthetic frame objects.
    """

    def __init__(
        self,
        path: str,
        *,
        only_downstream: bool = True,
        **kwargs,
    ) -> None:
        # Only forward kwargs to Pipecat's BaseObject when it is present;
        # ``object.__init__`` accepts no keyword arguments.
        if HAVE_PIPECAT:
            super().__init__(**kwargs)

        self._path = path
        self._only_downstream = only_downstream
        self._fh: Optional[TextIO] = open(path, "a", encoding="utf-8")

        self._turn_id = 0
        self._turn_open = False
        self._written: set[str] = set()

    # --- Pipecat observer hook ------------------------------------------

    async def on_push_frame(self, data) -> None:
        """Pipecat 1.5.x observer entry point (``FramePushed`` event data)."""
        if self._only_downstream:
            direction = getattr(data, "direction", None)
            # FrameDirection.DOWNSTREAM has ``.name == "DOWNSTREAM"``.
            if direction is not None and getattr(direction, "name", None) != "DOWNSTREAM":
                return
        frame = getattr(data, "frame", None)
        if frame is not None:
            self.handle_frame(frame)

    # --- Testable, pipecat-free core ------------------------------------

    def handle_frame(self, frame, t: Optional[float] = None) -> None:
        """Classify a frame and record the mapped event (once per turn).

        ``t`` is the event timestamp in epoch seconds; defaults to now. This
        method contains no Pipecat imports so it can be exercised with
        synthetic frame objects whose class name matches a mapped frame.
        """
        name = type(frame).__name__
        func_event = _FUNC_FRAME_EVENT.get(name)
        event = _FRAME_EVENT.get(name)
        if func_event is None and event is None:
            return
        if t is None:
            t = time.time()

        if func_event is not None:
            self._emit_function(func_event, frame, t)
            return

        if event == _START_EVENT:
            if self._turn_open and not self._written:
                # A second start frame for the same turn before any downstream
                # progress (e.g. a VAD + UserStopped pair) -> same turn.
                return
            # Either no turn is open, or the open turn has already progressed
            # and is now superseded by a barge-in. Abandon the previous
            # (incomplete) turn -- replay/stats will skip it -- and open a
            # fresh one so the new turn's events are never mis-attributed.
            self._turn_id += 1
            self._turn_open = True
            self._written = set()
            self._emit(_START_EVENT, t)
            return

        if not self._turn_open or event in self._written:
            return

        self._written.add(event)
        self._emit(event, t)

        if event == _END_EVENT:
            self._turn_open = False

    def _emit_function(self, event: str, frame, t: float) -> None:
        """Record a function/tool-call event with its name and call id.

        Ignored outside an open turn (a stray call with nowhere to attach). Does
        not touch ``self._written`` so it never counts as turn progress for the
        barge-in logic.
        """
        if not self._turn_open:
            return
        if self._fh is None:
            return
        line = json.dumps(
            {
                "turn_id": self._turn_id,
                "event": event,
                "t": round(t, 3),
                "name": getattr(frame, "function_name", None),
                "call_id": getattr(frame, "tool_call_id", None),
            }
        )
        self._fh.write(line + "\n")
        self._fh.flush()

    # --- Output ----------------------------------------------------------

    def _emit(self, event: str, t: float) -> None:
        if self._fh is None:
            return
        line = json.dumps(
            {"turn_id": self._turn_id, "event": event, "t": round(t, 3)}
        )
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying JSONL file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "VoxprofileObserver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
