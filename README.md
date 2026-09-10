# agent-claude

One of two agents built to compare **where data comes from**, holding everything else equal.
This one writes its own SQL and runs it read-only against Postgres. The other (agent-cortex) does the opposite.

Both run Claude Sonnet 5 through the same pipeline; the only intended difference is the
data-fetch step, so any cost or quality gap is attributable to that.

## Run it

```bash
cd apps/agent && .venv/bin/python -m uvicorn app.main:app --port 8001
```

Chat UI is served by the agent itself at <http://localhost:8001>. Allow ~40s to become
healthy — it front-loads context at boot, so an earlier health check is a false alarm.

See [RUNBOOK.md](RUNBOOK.md) first: it lists four things that silently disable observability
with no error message.

## Layout

| Path | |
|---|---|
| `apps/agent/app/` | the whole agent — FastAPI, LangGraph pipeline, chat UI |
| `apps/agent/app/mcp/client.py` | the data-path seam; the single `if` that separates the two variants |
| `apps/agent/app/static/index.html` | the chat page served at `/` |
| `apps/agent/app/resolver/` | SQL guard, tool loop, read-only DB access |
| `packages/ns_probe/` | observability SDK, installed editable — **must** be installed or traces vanish silently |

## Reading order

1. [DECISIONS.md](DECISIONS.md) — why it is built this way
2. [FLOW.md](FLOW.md) — one question, start to finish
3. [RUNBOOK.md](RUNBOOK.md) — running it, and the traps
