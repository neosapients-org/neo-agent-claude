"""Helpers for attaching rich, human-readable detail to the current ns_probe span.

Each graph node is wrapped with ``observe(name=...)`` (see app/graph/nodes/*.py),
which creates a nicely-named span but captures only code location by default. These
helpers let a node annotate ITS OWN span with the semantic data a reader wants in the
trace explorer — what the query was enriched to, what went into / came out of the
platform, the chosen plan, the final answer — via ``traceloop.entity.input/output``
(the fields the UI renders) plus namespaced ``ns.*`` attributes.

Fail-open by design: if ns_probe is absent or anything raises, annotation is a no-op
and never affects the agent.
"""
from __future__ import annotations

from typing import Any

try:
    from ns_probe import current_span, observe
except Exception:  # ns_probe not installed / disabled
    def current_span():  # type: ignore
        return None

    def observe(*a, **kw):  # type: ignore
        def _wrap(fn):
            return fn
        return _wrap


def _ser(value: Any) -> str:
    """Full, untruncated serialization (orjson → json → str), matching ns_probe."""
    if isinstance(value, str):
        return value
    try:
        import orjson
        return orjson.dumps(value, default=str, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    except Exception:
        try:
            import json
            return json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            return str(value)


def annotate(*, input: Any = None, output: Any = None, **attrs: Any) -> None:
    """Attach semantic input/output + ``ns.*`` attributes to the current span.

    Args:
        input:  value shown in the trace explorer's INPUT panel (traceloop.entity.input)
        output: value shown in the OUTPUT panel (traceloop.entity.output)
        **attrs: extra attributes; keys are stored verbatim (pass fully-qualified,
                 e.g. ``ns_intent="..."`` → attribute ``ns.intent``). Non-scalars are
                 JSON-serialized.
    """
    span = current_span()
    if span is None:
        return
    try:
        if input is not None:
            span.set_attribute("traceloop.entity.input", _ser(input))
        if output is not None:
            span.set_attribute("traceloop.entity.output", _ser(output))
        for key, val in attrs.items():
            # allow ns_foo kwarg → "ns.foo" attribute (kwargs can't contain dots)
            akey = key.replace("ns_", "ns.", 1) if key.startswith("ns_") else key
            span.set_attribute(akey, val if isinstance(val, (str, int, float, bool)) else _ser(val))
    except Exception:
        pass


def observed_node(fn, *, name: str, agent_name: str = "agent", detail=None):
    """Wrap an async graph node with an ``observe`` span AND semantic annotation.

    Replaces the bare ``observe(name=...)(fn)`` decoration. After the node runs,
    ``detail(state, result)`` (if given) returns a dict of ``annotate()`` kwargs
    (``input=``, ``output=``, ``ns_*=``) describing what the step actually did.
    Fail-open: annotation errors never propagate.
    """
    import functools

    @functools.wraps(fn)
    async def _wrapper(state):
        result = await fn(state)
        if detail is not None:
            try:
                kwargs = detail(state, result) or {}
                annotate(**kwargs)
            except Exception:
                pass
        return result

    return observe(name=name, agent_name=agent_name)(_wrapper)
