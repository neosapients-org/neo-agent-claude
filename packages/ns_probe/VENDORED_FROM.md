# Where this copy came from

`ns_probe/` here is a vendored copy, not the canonical source. Do not fix bugs here first —
fix them upstream and re-vendor, or the two drift apart silently.

| | |
|---|---|
| **Upstream repo** | `local/neo-platform` |
| **Branch** | `ns-probe-oss` |
| **Path** | `packages/python/ns-probe/` |
| **Commit** | `09b59c7e1df62703bf92064f9c2f1f693516949e` |
| **Vendored** | 2026-09-08 |

## ⚠️ This copy is ahead of that commit

It also contains an **uncommitted** local fix to the Anthropic instrumentor
(`ns_probe/instrumentors.py`), which is not in the SHA above. Until that fix is committed
and the branch pushed, this vendored tree cannot be reproduced from git by anyone else.

The fix closes three holes that each caused Claude calls to record **zero tokens** — and
because cost is computed *from* tokens, a zero reads as a cheap model rather than a broken
probe. Nothing errors when it fails:

1. Only the synchronous `Messages.create` was wrapped, not `AsyncMessages.create`. Any async
   agent recorded nothing at all.
2. No streaming path. A streamed call returns an iterator, not an object with `.usage`, so
   reading usage off it captured nothing.
3. `cache_read_input_tokens` / `cache_creation_input_tokens` were ignored, so with prompt
   caching on, the recorded input count collapsed to a fraction of what was billed.

Covered by `tests/test_anthropic_instrumentor.py` (12 tests).

## Why the tests are vendored too

`tests/` is copied alongside the source deliberately, so this repo can prove its own copy of
the probe works rather than trusting that an upstream CI run covered it. Run them with:

```bash
python -m pytest packages/ns_probe/tests -q
```

`test_skill_is_current.py` is deliberately **not** vendored: it validates upstream's `skill/`
documentation directory and its generator script — repo assets, not library behaviour — and
fails with 8 collection errors without them. Result here: **183 passed, 3 skipped** (the skips
are optional deps: `opentelemetry`, `cryptography`).

## Deviations from upstream

Three:

- Build backend stays `hatchling` and the package name stays `ns_probe`, matching how this
  repo installs it (`pip install -e packages/ns_probe`). Upstream builds a differently-named
  distribution for publication.
- `requires-python` raised to `>=3.10`. The source uses `X | Y` type syntax at runtime, which
  raises on 3.9 at import time.
