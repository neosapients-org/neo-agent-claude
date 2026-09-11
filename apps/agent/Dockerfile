FROM python:3.12-slim

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all packages. NOTE: build context is the repository root, so paths are
# relative to the monorepo layout (packages/, apps/).
COPY packages/ns_probe/ /workspace/ns_probe/
COPY packages/neo_guardrail_hub/ /workspace/neo_guardrail_hub/
COPY packages/neohub-memory/ /workspace/neohub-memory/
COPY apps/agent/ /workspace/agent/

# Install vim
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to prevent CUDA packages being pulled in by llm-guard
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (split for easier debugging)
# anthropic is NOT optional despite LLM_PROVIDER defaulting to openai: app/llm.py
# imports AsyncAnthropic at module scope, so leaving it out makes main.py die at
# boot with ModuleNotFoundError. It is listed in requirements.txt, but this
# Dockerfile installs an explicit list rather than that file (requirements.txt
# ends in `-e ./packages/ns_probe`, a repo-root-relative path that does not exist
# at this layout), so the two lists have to be kept in step by hand.
RUN pip install --no-cache-dir fastapi uvicorn langchain-openai langchain-core langgraph \
    pydantic python-dotenv mcp openai qdrant-client anthropic
# The direct resolver's two hard dependencies, and the reason this variant could not
# read the database at all: asyncpg opens the read-only Neon connection, sqlglot parses
# the model's SQL before it is executed (app/resolver/sql_guard.py). Both are declared in
# requirements.txt, which this Dockerfile does not install — so in the image the import
# in app/resolver/direct.py raised ModuleNotFoundError, the resolve_context interception
# in app/mcp/client.py failed, and every data question came back as "no rows found"
# rather than as an error naming a missing package.
RUN pip install --no-cache-dir "sqlglot>=25.0.0" "asyncpg>=0.29.0"
# ns_probe is now an in-tree editable package (packages/ns_probe). Its
# instrumentors already emit traceloop.entity.input/output and human-readable
# span names + full per-node input/output — no post-install patching needed.
RUN pip install --no-cache-dir -e "/workspace/ns_probe"
# [nemo] adds nemoguardrails + langchain so nemo_self_check_* and the topical/domain rail run
RUN pip install --no-cache-dir -e "/workspace/neo_guardrail_hub[llm-guard,nemo]"
# optimum[onnxruntime] enables LLM Guard's ONNX scanners (relevance, toxicity_output,
# etc.) — without it they log "No module named 'optimum'" and pass vacuously.
RUN pip install --no-cache-dir "optimum[onnxruntime]"
RUN pip install --no-cache-dir -e "/workspace/neohub-memory"

WORKDIR /workspace/agent

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
