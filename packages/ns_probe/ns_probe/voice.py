"""
Voice-agent tracing for ns_probe (LiveKit `AgentSession`).

A voice agent's most important numbers live inside the audio pipeline — how long
speech recognition took, how long the agent waited before deciding the user had
finished, and how quickly the first byte of audio came back. None of that is
visible from the agent's own code: STT, end-of-turn detection and TTS all run
inside LiveKit's `AgentSession`, so there is no function of yours to wrap.

LiveKit already measures all of it and emits `metrics_collected` when each
operation completes. Subscribing once is far cheaper and less brittle than
monkey-patching the STT/TTS plugins, and it is the only place TTS
time-to-first-byte is exposed at all.

    from ns_probe.voice import instrument_livekit, voice_call, voice_turn

    instrument_livekit(session)                      # STT / EOU / TTS spans

    with voice_call(session_id=sid, user_id=uid) as call:
        instrument_livekit(session, call=call)
        while talking:
            with voice_turn(call) as turn:
                turn.agent_said(question)            # span input
                reply = await listen()
                turn.user_said(reply)                # span output

Metrics are reported *after* the operation finishes, so each span is written
with its real start/end timestamps — otherwise every audio span would collapse
onto the moment the notification arrived instead of sitting where it belongs in
the waterfall.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .span import Span
from .spans import add_event, end_span, record_span, set_span_io, start_span

logger = logging.getLogger("ns_probe.voice")

__all__ = ["instrument_livekit", "voice_call", "voice_turn", "VoiceCall", "VoiceTurn"]

_NS_PER_SEC = 1_000_000_000


class VoiceCall:
    """One voice session. Holds the root span every turn hangs off."""

    __slots__ = ("span", "_turns")

    def __init__(self, span: Optional[Span]) -> None:
        self.span = span
        self._turns = 0

    @property
    def turns(self) -> int:
        return self._turns

    def _next_turn_index(self) -> int:
        self._turns += 1
        return self._turns

    def current_parent(self) -> Optional[Span]:
        return self.span


class VoiceTurn:
    """One exchange: what the agent said, and what the user said back.

    Consecutive agent lines are appended rather than starting a new turn — an
    agent often speaks an acknowledgement and then the next question before
    waiting, and that is one exchange from the user's point of view.
    """

    __slots__ = ("span", "index", "_agent_lines", "_started")

    def __init__(self, span: Optional[Span], index: int) -> None:
        self.span = span
        self.index = index
        self._agent_lines: list = []
        self._started = time.monotonic()

    def agent_said(self, text: str) -> None:
        """Record a line the agent spoke (call once per line)."""
        if text:
            self._agent_lines.append(text)
        set_span_io(self.span, inputs={"agent_said": " ".join(self._agent_lines)})

    def user_said(self, text: str, source: Optional[str] = None) -> None:
        """Record the user's reply. `source` notes how it arrived (voice/text/…)."""
        payload: dict = {"user_said": text}
        if source:
            payload["source"] = source
        set_span_io(self.span, outputs=payload)

    @property
    def agent_lines(self) -> int:
        return len(self._agent_lines)


def _attach_call(session: Any, call: Optional[VoiceCall]) -> None:
    """Remember the call on the session so the metrics hook can find it."""
    try:
        setattr(session, "_ns_probe_voice_call", call)
    except Exception:
        pass


def instrument_livekit(
    session: Any,
    call: Optional["VoiceCall"] = None,
    parent: Optional[Any] = None,
) -> bool:
    """Turn a LiveKit ``AgentSession``'s metrics into spans. Call once per session.

    Produces, per operation, a span parented to the turn currently open on
    `call` (or to the call itself between turns):

    ============================  ==================================================
    Transcribe user speech        audio duration -> request duration
    Wait for the user to finish   -> end-of-utterance delay, transcription delay
    Speak the reply (TTS)         characters -> time-to-first-byte, audio duration
    ============================  ==================================================

    A cancelled utterance also emits a "User interrupted the agent" event —
    barge-in, which is otherwise buried as a boolean on the TTS span.

    Parenting, in order of precedence:
      * `parent` — a zero-argument callable returning the span to attach to.
        Use this when turns are not managed by :func:`voice_turn`, e.g. an agent
        whose turn opens in one method and closes in another::

            instrument_livekit(session, parent=lambda: channel.current_span())

      * `call` — a :class:`VoiceCall`; spans attach to its open turn, or to the
        call itself between turns.
      * neither — spans become roots.

    Returns True if the hook was attached. Never raises: a session that cannot
    be instrumented degrades to no voice spans, not a broken call.
    """
    if parent is not None:
        try:
            session._ns_probe_voice_parent = parent
        except Exception:
            pass
    if getattr(session, "_ns_probe_voice_instrumented", False):
        _attach_call(session, call)
        return True
    _attach_call(session, call)
    try:

        @session.on("metrics_collected")
        def _on_metrics(event: Any) -> None:  # noqa: ANN001 - LiveKit event object
            try:
                _record_metric(session, getattr(event, "metrics", None))
            except Exception:  # noqa: BLE001 - telemetry must never break a call
                pass

        session._ns_probe_voice_instrumented = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not attach metrics hook: %s", exc)
        return False


def _parent_for(session: Any) -> Optional[Span]:
    """Span the audio metrics should attach to: caller's callable, else the turn, else the call."""
    provider = getattr(session, "_ns_probe_voice_parent", None)
    if provider is not None:
        try:
            span = provider()
            if span is not None:
                return span
        except Exception:
            pass  # fall through to the call/turn tracking below
    call: Optional[VoiceCall] = getattr(session, "_ns_probe_voice_call", None)
    if call is None:
        return None
    turn: Optional[VoiceTurn] = getattr(session, "_ns_probe_voice_turn", None)
    if turn is not None and turn.span is not None:
        return turn.span
    return call.span


def _window(timestamp: Optional[float], duration_s: Optional[float]):
    """(start_ns, end_ns) for an operation that finished at `timestamp`."""
    if not timestamp:
        return None, None
    end_ns = int(timestamp * _NS_PER_SEC)
    if not duration_s:
        return end_ns, end_ns
    return end_ns - int(duration_s * _NS_PER_SEC), end_ns


def _record_metric(session: Any, metric: Any) -> None:
    if metric is None:
        return
    kind = type(metric).__name__
    parent = _parent_for(session)
    timestamp = getattr(metric, "timestamp", None)
    get = lambda name: getattr(metric, name, None)  # noqa: E731

    if kind == "STTMetrics":
        start_ns, end_ns = _window(timestamp, get("duration"))
        record_span(
            "Transcribe user speech",
            parent=parent,
            inputs={"audio_duration_s": get("audio_duration"), "streamed": get("streamed")},
            outputs={"request_duration_s": get("duration")},
            attributes={"voice.stage": "stt", "voice.provider": get("label")},
            start_time_ns=start_ns,
            end_time_ns=end_ns,
        )

    elif kind == "EOUMetrics":
        # The wait between "user stopped talking" and "we decided they were
        # done" — the dial between cutting people off and feeling sluggish.
        delay = get("end_of_utterance_delay")
        start_ns, end_ns = _window(timestamp, delay)
        record_span(
            "Wait for the user to finish",
            parent=parent,
            inputs={"speech_id": get("speech_id")},
            outputs={
                "end_of_utterance_delay_s": delay,
                "transcription_delay_s": get("transcription_delay"),
                "on_user_turn_completed_delay_s": get("on_user_turn_completed_delay"),
            },
            attributes={"voice.stage": "end_of_turn"},
            start_time_ns=start_ns,
            end_time_ns=end_ns,
        )

    elif kind == "TTSMetrics":
        start_ns, end_ns = _window(timestamp, get("duration"))
        cancelled = bool(get("cancelled"))
        record_span(
            "Speak the reply (TTS)",
            parent=parent,
            inputs={"characters_count": get("characters_count"), "streamed": get("streamed")},
            outputs={
                "ttfb_s": get("ttfb"),
                "audio_duration_s": get("audio_duration"),
                "cancelled": cancelled,
            },
            attributes={
                "voice.stage": "tts",
                "voice.provider": get("label"),
                "voice.tts.cancelled": cancelled,
            },
            start_time_ns=start_ns,
            end_time_ns=end_ns,
        )
        if cancelled:
            # The agent was cut off mid-utterance: the user talked over it.
            add_event(
                parent,
                "User interrupted the agent",
                attributes={
                    "event.type": "barge_in",
                    "spoken_chars_before_cut": get("characters_count"),
                },
            )


class _CallCtx:
    """Context manager returned by :func:`voice_call`."""

    __slots__ = ("_call", "_session", "_attrs")

    def __init__(self, call: VoiceCall, session: Any, attrs: dict) -> None:
        self._call = call
        self._session = session
        self._attrs = attrs

    def __enter__(self) -> VoiceCall:
        return self._call

    def __exit__(self, exc_type, exc, tb) -> bool:
        set_span_io(
            self._call.span,
            outputs={"turns": self._call.turns, **{k: v for k, v in self._attrs.items()}},
        )
        end_span(
            self._call.span,
            attributes={"voice.call.turns": self._call.turns},
            error=exc if exc is not None else None,
        )
        return False


def voice_call(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session: Any = None,
    name: str = "Voice call",
    attributes: Optional[dict] = None,
    **outputs: Any,
) -> _CallCtx:
    """Open the root span for a voice session.

    Everything else — the graph, the LLM calls, each turn and its audio spans —
    hangs off this, so a trace is identifiable as *this call for this user*
    rather than an anonymous pipeline. Pass `session` to let
    :func:`instrument_livekit` find the call automatically.
    """
    span = start_span(
        name,
        attributes={
            "voice.session.id": session_id,
            "voice.user.id": user_id,
            **(attributes or {}),
        },
    )
    set_span_io(span, inputs={"session_id": session_id, "user_id": user_id})
    call = VoiceCall(span)
    if session is not None:
        _attach_call(session, call)
    return _CallCtx(call, session, outputs)


class _TurnCtx:
    """Context manager returned by :func:`voice_turn`."""

    __slots__ = ("_turn", "_session")

    def __init__(self, turn: VoiceTurn, session: Any) -> None:
        self._turn = turn
        self._session = session

    def __enter__(self) -> VoiceTurn:
        return self._turn

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = round((time.monotonic() - self._turn._started) * 1000, 1)
        end_span(
            self._turn.span,
            attributes={
                "voice.turn.index": self._turn.index,
                "voice.turn.duration_ms": duration_ms,
                # >1 means the agent spoke several lines before waiting.
                "voice.turn.agent_lines": self._turn.agent_lines or None,
            },
            error=exc if exc is not None else None,
        )
        if self._session is not None:
            try:
                self._session._ns_probe_voice_turn = None
            except Exception:
                pass
        return False


def voice_turn(
    call: VoiceCall,
    session: Any = None,
    name: str = "Conversation turn",
) -> _TurnCtx:
    """Open a span for one exchange, parented to `call`.

    Pass `session` (or the one already given to :func:`voice_call`) so the audio
    metric spans for this exchange nest inside the turn rather than the call.
    """
    index = call._next_turn_index()
    span = start_span(name, attributes={"voice.turn.index": index}, parent=call.span)
    turn = VoiceTurn(span, index)
    target = session if session is not None else None
    if target is not None:
        try:
            target._ns_probe_voice_turn = turn
        except Exception:
            pass
    return _TurnCtx(turn, target)
