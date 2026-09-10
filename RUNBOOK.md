# Running the experiment locally

Everything below was working as of 2026-09-09. Ports are chosen to avoid the platform's
own (3000/3001/3002), so all of it can run side by side.

## The two agents

```bash
# agent-cortex
cd agent-cortex/apps/agent && .venv/bin/python -m uvicorn app.main:app --port 8000
# agent-claude
cd agent-claude/apps/agent && .venv/bin/python -m uvicorn app.main:app --port 8001
```

Chat UI is served by the agent itself at `/`. Takes ~40s to become healthy — it front-loads
context at boot, so a health check before then is a false alarm.

**Four things that silently break observability if wrong.** Each one fails without an error:

| | Why it matters |
|---|---|
| `pip install -e packages/ns_probe` | `main.py` catches ImportError and sets `NS_PROBE_ENABLED = False`. Missing install = no traces at all, no warning. It is in `requirements.txt` now. |
| `load_dotenv()` before `configure()` | `configure()` reads the endpoint and service name from the ENVIRONMENT. `app.config` loads `.env` ~40 lines later, so without the early load the agent used built-in defaults. Only ever showed up outside Docker. |
| `NS_PROBE_SERVICE_NAME` | Splits the two variants apart on the dashboard. |
| `OTEL_SERVICE_NAME` (same value) | What DW Foundry's "Observability ID" matches on. If it drifts from the registered worker, traces just stop appearing under it. |

Check with `curl localhost:8001/debug/ns_probe` — it reports `enabled`, the endpoint, and
export counts. `{"enabled": false}` means the ns_probe import failed.

## Observability stack (Docker)

```bash
cd neo-platform/ops/compose && docker compose -p cml up -d
```

Collector **:4328**, processor :8010, API :8087, ClickHouse :8123.

Anthropic pricing lives on branch `fix/anthropic-pricing` in the `neo-platform-develop`
worktree. Without it every Claude call is priced at **$0.00** — `calculate_cost()` returns
0.0 for an unknown model with only a debug log.

```bash
# rebuild after changing pricing
docker build -f ops/docker/services/neo-observe-processor.Dockerfile -t neo-observe-processor:latest .
docker compose -p cml up -d --no-deps --force-recreate neo-observe-processor
```

## Reading the numbers

The UI is optional; the data is queryable directly.

```bash
curl -s 'http://localhost:8123/' --data-binary "
SELECT service_name, count() AS spans, sum(tokens_input) AS tok_in,
       sum(tokens_output) AS tok_out, round(sum(cost_usd), 5) AS cost_usd
FROM observability.traces
WHERE ingested_at > now() - INTERVAL 60 MINUTE
GROUP BY service_name FORMAT PrettyCompactMonoBlock"
```

Per-turn summary rows are in `observability.outcome_ledger`, keyed by `agent_id`
(= the service name). Note the time column is `ingested_at` / `start_time`, not `timestamp`.

## Control plane (only needed for the DW Foundry UI)

```bash
docker run -d --name neo-postgres --network neo-network -p 5433:5432 \
  -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres postgres:17-alpine
docker run -d --name neo-redis --network neo-network -p 6380:6379 redis:8-alpine
docker run -d --name neo-mongodb --network neo-network -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=password mongo:8.0

docker exec neo-postgres psql -U admin -d postgres -c "CREATE DATABASE cml_platform;"
DATABASE_URL="postgresql://admin:password@localhost:5433/cml_platform" \
  npx prisma migrate deploy --schema=packages/backend/database/prisma/schema.prisma

npx nx serve neo-auth-svc      # gRPC 50055
npx nx serve neo-tenant-svc    # gRPC 50053
npx nx serve neo-worker-svc    # gRPC 50058
npx nx serve neo-cp-gateway-svc  # :3001
npx nx serve neo-gateway-svc     # :3000  <- the workers API lives here, not on the CP gateway
cd services/neo-cp-frontend && npm run dev   # :3002
```

Config gaps found doing this, all of which only appear outside Docker:

- `neo-gateway-svc/.env` did not exist — needs `DATABASE_URL`, `MONGODB_URI`, `PORT=3000`.
- `neo-worker-svc`'s in-code Mongo fallback omits `?authSource=admin`, so it authenticates
  against the wrong database and fails.
- `docker-compose.control-plane.yml` declares `depends_on: neo-postgres/neo-redis/neo-mongodb`
  — those are CONTAINER names; the services in the infra file are `postgres/redis/mongodb`.
  Loading both files together fails with "invalid compose project".
- Restarting a service without freeing its port first races the old process and the new one
  dies on `EADDRINUSE`, leaving nothing listening. Symptom in the browser is "Failed to
  fetch", which looks like CORS and is not.

## Known blocker: registering workers in DW Foundry

Not solvable locally. `POST /v1/workers` needs a tenant-scoped JWT, and the gateway verifies
it against Scalekit's **remote JWKS** — there is no local secret or dev verification path, so
a token can only come from Scalekit.

The dev auth bypass (`NEXT_PUBLIC_DEV_AUTH_BYPASS=true`, wired into `AuthContext`,
`app/page.tsx`, `WorkspaceContext` and `auth-interceptor`) gets you into the UI, but it
produces no session — so every gateway call returns 401.

Real sign-in additionally needs a tenant, and `prisma:seed:platform` requires
`SCALEKIT_ENV_URL` + `PLATFORM_ADMIN_EMAILS`. **That seed provisions a real organisation and
real users in live Scalekit** — its own docstring warns about this. It is a deliberate
decision, not a setup step.

**None of this blocks the experiment.** Trace and cost data is complete and queryable; DW
Foundry is one view of it.
