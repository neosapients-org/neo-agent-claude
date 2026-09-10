"""Full-chain observability for every answer.

Emits one structured JSON log line per stage of the data path so any answer can be
diffed end-to-end and errors attributed to the agent vs the platform:

    user query -> enriched query -> exact MCP query -> raw platform payload
               -> parsed data -> final answer   (+ retry count, per-step latency)

These are plain `logging` records on the "agent.chain" logger (level INFO), prefixed
with "CHAIN " and carrying a compact JSON body, so they are easy to grep/ship:

    docker logs <agent> | grep '^CHAIN ' | sed 's/^CHAIN //' | jq .

Correlation: records share a `cid` (correlation id) derived from the user query +
user id, so the fetch stage and the answer stage of the same turn line up.
"""
import hashlib
import json
import logging

_logger = logging.getLogger("agent.chain")

# Keep payloads readable in logs but bounded so we never dump 60k of data per line.
_MAX_FIELD = 800


def make_cid(user_query: str, user_id: str = "") -> str:
    """Stable short correlation id for a turn (same query+user => same cid)."""
    h = hashlib.sha1(f"{user_id}|{user_query}".encode("utf-8", "ignore")).hexdigest()
    return h[:10]


def _clip(v):
    if isinstance(v, str) and len(v) > _MAX_FIELD:
        return v[:_MAX_FIELD] + f"...(+{len(v) - _MAX_FIELD} chars)"
    return v


def log_chain(stage: str, cid: str = "", **fields) -> None:
    """Emit one chain record. Never raises — observability must not break the agent."""
    try:
        body = {"cid": cid, "stage": stage, **{k: _clip(v) for k, v in fields.items()}}
        _logger.info("CHAIN %s", json.dumps(body, default=str, ensure_ascii=False))
    except Exception:  # pragma: no cover - logging must never break the request
        pass
