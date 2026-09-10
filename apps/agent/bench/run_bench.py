"""
agent Agent — Latency & Accuracy Benchmark Runner
=====================================================
Runs all queries from queries.json against the agent /chat endpoint.
Outputs: bench_results.json (raw) + bench_report.md (formatted report).

Usage:
    python -m bench.run_bench                         # default: localhost:8000
    python -m bench.run_bench --base-url http://host:port
    python -m bench.run_bench --skip-warmup           # skip warmup query
"""

import json
import time
import asyncio
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone

import httpx

BENCH_DIR = Path(__file__).parent
QUERIES_FILE = BENCH_DIR / "queries.json"
RESULTS_FILE = BENCH_DIR / "bench_results.json"
REPORT_FILE = BENCH_DIR / "bench_report.md"

DEFAULT_BASE_URL = "http://localhost:8000"
USER_ID = "bench_eval_user"


async def call_agent(client: httpx.AsyncClient, base_url: str, query: str, query_id: str = "") -> dict:
    """Call /chat and return response + latency."""
    # Use unique user_id per query to prevent session cross-contamination
    user_id = f"bench_{query_id}_{int(time.time())}" if query_id else USER_ID
    payload = {"message": query, "user_id": user_id}
    start = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/chat", json=payload, timeout=120.0)
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "response": data.get("response", ""),
                "latency_breakdown": data.get("latency_breakdown", {}),
                "total_ms": data.get("latency_breakdown", {}).get("total_ms", elapsed_ms),
                "http_ms": elapsed_ms,
                "mcp_tool_calls": data.get("mcp_tool_calls", []),
            }
        else:
            return {
                "success": False,
                "response": f"HTTP {resp.status_code}: {resp.text[:500]}",
                "latency_breakdown": {},
                "total_ms": elapsed_ms,
                "http_ms": elapsed_ms,
                "mcp_tool_calls": [],
            }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        return {
            "success": False,
            "response": f"ERROR: {e}",
            "latency_breakdown": {},
            "total_ms": elapsed_ms,
            "http_ms": elapsed_ms,
            "mcp_tool_calls": [],
        }


def compute_latency_stats(results: list[dict]) -> dict:
    """Compute latency percentiles grouped by scenario."""
    # Overall
    all_ms = [r["total_ms"] for r in results if r["success"]]
    overall = _percentiles(all_ms) if all_ms else {}

    # Per scenario
    scenarios: dict[str, list[float]] = {}
    for r in results:
        if r["success"]:
            sc = r["scenario"]
            scenarios.setdefault(sc, []).append(r["total_ms"])

    per_scenario = {sc: _percentiles(vals) for sc, vals in scenarios.items()}

    # Per stage (tool_selection_ms, mcp_exec_ms, etc.)
    stage_latencies: dict[str, list[float]] = {}
    for r in results:
        if r["success"]:
            for k, v in r.get("latency_breakdown", {}).items():
                if k != "total_ms" and isinstance(v, (int, float)):
                    stage_latencies.setdefault(k, []).append(v)
    per_stage = {k: _percentiles(v) for k, v in stage_latencies.items()}

    return {"overall": overall, "per_scenario": per_scenario, "per_stage": per_stage}


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "min": round(s[0], 1),
        "p50": round(statistics.median(s), 1),
        "p75": round(s[int(n * 0.75)] if n > 1 else s[0], 1),
        "p95": round(s[int(n * 0.95)] if n > 2 else s[-1], 1),
        "p99": round(s[min(int(n * 0.99), n - 1)], 1),
        "max": round(s[-1], 1),
        "mean": round(statistics.mean(s), 1),
        "stdev": round(statistics.stdev(s), 1) if n > 1 else 0,
    }


def generate_report(results: list[dict], stats: dict, cold_start_ms: float | None) -> str:
    """Generate bench_report.md content."""
    lines = [
        "# agent Agent — Benchmark Report",
        f"\n**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Queries:** {len(results)}",
        f"**Success Rate:** {sum(1 for r in results if r['success'])}/{len(results)}",
    ]
    if cold_start_ms:
        lines.append(f"**Cold Start (first query):** {cold_start_ms:.0f} ms")

    # Overall latency
    lines.append("\n## Overall Latency")
    if stats["overall"]:
        o = stats["overall"]
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k in ["count", "min", "p50", "p75", "p95", "p99", "max", "mean", "stdev"]:
            lines.append(f"| {k} | {o[k]} ms |")

    # Per scenario
    lines.append("\n## Latency by Scenario")
    lines.append("| Scenario | Count | p50 | p75 | p95 | Mean | Max |")
    lines.append("|----------|-------|-----|-----|-----|------|-----|")
    for sc, st in sorted(stats["per_scenario"].items()):
        lines.append(f"| {sc} | {st['count']} | {st['p50']} | {st['p75']} | {st['p95']} | {st['mean']} | {st['max']} |")

    # Per stage
    lines.append("\n## Latency by Stage")
    lines.append("| Stage | Count | p50 | p75 | p95 | Mean |")
    lines.append("|-------|-------|-----|-----|-----|------|")
    for st, vals in sorted(stats["per_stage"].items()):
        lines.append(f"| {st} | {vals['count']} | {vals['p50']} | {vals['p75']} | {vals['p95']} | {vals['mean']} |")

    # Individual results table
    lines.append("\n## Individual Query Results")
    lines.append("| # | ID | Scenario | Total ms | Tool Sel ms | MCP ms | Success |")
    lines.append("|---|-----|----------|----------|-------------|--------|---------|")
    for i, r in enumerate(results, 1):
        lb = r.get("latency_breakdown", {})
        ts = lb.get("tool_selection_ms", "-")
        mc = lb.get("mcp_exec_ms", "-")
        lines.append(f"| {i} | {r['id']} | {r['scenario']} | {r['total_ms']} | {ts} | {mc} | {'✓' if r['success'] else '✗'} |")

    # Response column (for accuracy comparison later)
    lines.append("\n## Agent Responses (for accuracy comparison)")
    lines.append("| ID | Query | Agent Response | Platform Response (paste here) | Accuracy Score |")
    lines.append("|----|-------|----------------|-------------------------------|----------------|")
    for r in results:
        # Truncate response for readability, full data in JSON
        resp_short = r["response"][:200].replace("\n", " ").replace("|", "\\|") if r["response"] else "-"
        lines.append(f"| {r['id']} | {r['query'][:60]}... | {resp_short} | | |")

    return "\n".join(lines)


async def run_benchmark(base_url: str, skip_warmup: bool = False):
    """Main benchmark runner."""
    queries = json.loads(QUERIES_FILE.read_text())
    print(f"[BENCH] Loaded {len(queries)} queries")
    print(f"[BENCH] Target: {base_url}/chat")
    print(f"[BENCH] User ID: {USER_ID}")
    print()

    results = []
    cold_start_ms = None

    async with httpx.AsyncClient() as client:
        # --- Warmup ---
        if not skip_warmup:
            print("[BENCH] Running warmup query (cold start)...")
            warmup_result = await call_agent(client, base_url, "hello")
            cold_start_ms = warmup_result["total_ms"]
            print(f"[BENCH] Cold start: {cold_start_ms} ms")
            # Brief pause to let things settle
            await asyncio.sleep(1)

        # --- Main run ---
        for i, q in enumerate(queries, 1):
            qid = q["id"]
            query = q["query"]
            scenario = q["scenario"]

            print(f"[{i:02d}/{len(queries)}] {qid} ({scenario}) — {query[:60]}...")

            result = await call_agent(client, base_url, query, query_id=qid)
            result["id"] = qid
            result["query"] = query
            result["scenario"] = scenario
            result["timestamp"] = datetime.now(timezone.utc).isoformat()

            status = "✓" if result["success"] else "✗"
            print(f"         {status} {result['total_ms']} ms | response: {result['response'][:80]}...")
            print()

            results.append(result)

            # Small delay between queries to avoid overloading
            await asyncio.sleep(0.5)

    # --- Stats ---
    stats = compute_latency_stats(results)

    # --- Save raw results ---
    output = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "total_queries": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "cold_start_ms": cold_start_ms,
        },
        "stats": stats,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n[BENCH] Raw results saved → {RESULTS_FILE}")

    # --- Generate report ---
    report = generate_report(results, stats, cold_start_ms)
    REPORT_FILE.write_text(report)
    print(f"[BENCH] Report saved → {REPORT_FILE}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)
    if stats["overall"]:
        o = stats["overall"]
        print(f"  Total queries:  {o['count']}")
        print(f"  Mean:           {o['mean']} ms")
        print(f"  Median (p50):   {o['p50']} ms")
        print(f"  p75:            {o['p75']} ms")
        print(f"  p95:            {o['p95']} ms")
        print(f"  Max:            {o['max']} ms")
    if cold_start_ms:
        print(f"  Cold start:     {cold_start_ms} ms")
    print()
    print("Per scenario:")
    for sc, st in sorted(stats["per_scenario"].items()):
        print(f"  {sc:15s}  p50={st['p50']:>6.0f}  p95={st['p95']:>6.0f}  mean={st['mean']:>6.0f}  n={st['count']}")


def main():
    parser = argparse.ArgumentParser(description="agent Agent Benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Agent base URL")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup query")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.base_url, args.skip_warmup))


if __name__ == "__main__":
    main()
