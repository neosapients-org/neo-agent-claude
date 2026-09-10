"""
agent Agent — Accuracy Evaluator
=====================================
Compares agent responses against platform ground-truth responses.
Uses LLM-as-judge to score: field_coverage, factual_accuracy, hallucination.

Usage:
    # After running run_bench.py and filling platform responses:
    python -m bench.accuracy_eval
    python -m bench.accuracy_eval --results bench_results.json --ground-truth ground_truth.json
"""

import json
import asyncio
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import httpx

BENCH_DIR = Path(__file__).parent
RESULTS_FILE = BENCH_DIR / "bench_results.json"
GROUND_TRUTH_FILE = BENCH_DIR / "ground_truth.json"
ACCURACY_REPORT_FILE = BENCH_DIR / "accuracy_report.md"
ACCURACY_RESULTS_FILE = BENCH_DIR / "accuracy_results.json"

# LLM Judge config — uses same OpenAI endpoint as the agent
import os
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4-nano")

JUDGE_SYSTEM_PROMPT = """You are an accuracy evaluator for a wealth management AI agent.
Compare the agent's response against the reference (platform) response for the given query.

Score each dimension from 0 to 10:

1. **field_coverage** (0-10): Does the agent mention the same key data fields as the reference?
   - 10 = all fields present, 0 = completely missing data
   - Partial = proportional score

2. **factual_accuracy** (0-10): Are the numbers, names, and facts correct when compared to reference?
   - 10 = all facts match exactly, 0 = completely wrong numbers/names
   - Minor rounding differences = 9

3. **hallucination** (0-10): Did the agent invent facts NOT in the reference?
   - 10 = no hallucination at all, 0 = entirely fabricated response
   - Any invented investor name or number = severe penalty

4. **relevance** (0-10): Does the response actually answer what was asked?
   - 10 = directly answers the question with the right data
   - 0 = completely off-topic or generic response

Return ONLY valid JSON (no markdown fences):
{"field_coverage": N, "factual_accuracy": N, "hallucination": N, "relevance": N, "reason": "brief explanation"}
"""

JUDGE_USER_TEMPLATE = """Query: {query}

Reference (Platform Response):
{reference}

Agent Response:
{agent_response}

Score the agent response against the reference."""


async def judge_response(
    client: httpx.AsyncClient,
    query: str,
    agent_response: str,
    reference: str,
) -> dict:
    """Use LLM to judge accuracy of agent response vs reference."""
    if not reference or not reference.strip():
        return {"field_coverage": -1, "factual_accuracy": -1, "hallucination": -1, "relevance": -1, "reason": "No reference provided"}

    if not agent_response or not agent_response.strip():
        return {"field_coverage": 0, "factual_accuracy": 0, "hallucination": 10, "relevance": 0, "reason": "Agent returned empty response"}

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            query=query,
            reference=reference,
            agent_response=agent_response,
        )},
    ]

    try:
        resp = await client.post(
            f"{JUDGE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
            json={"model": JUDGE_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": 300},
            timeout=60.0,
        )
        if resp.status_code != 200:
            return {"field_coverage": -1, "factual_accuracy": -1, "hallucination": -1, "relevance": -1, "reason": f"Judge API error: {resp.status_code}"}

        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        return {"field_coverage": -1, "factual_accuracy": -1, "hallucination": -1, "relevance": -1, "reason": f"Judge error: {e}"}


def compute_accuracy_stats(scores: list[dict]) -> dict:
    """Compute accuracy statistics across all judged queries."""
    # Filter out un-judged (score = -1)
    valid = [s for s in scores if s.get("field_coverage", -1) >= 0]
    if not valid:
        return {"message": "No valid scores to compute stats"}

    dims = ["field_coverage", "factual_accuracy", "hallucination", "relevance"]
    stats = {}

    for dim in dims:
        values = [s[dim] for s in valid if s.get(dim, -1) >= 0]
        if values:
            stats[dim] = {
                "count": len(values),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "mode": Counter(values).most_common(1)[0][0] if values else None,
                "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }

    # Composite accuracy score (weighted average)
    composite_scores = []
    for s in valid:
        fc = s.get("field_coverage", 0)
        fa = s.get("factual_accuracy", 0)
        ha = s.get("hallucination", 0)
        re = s.get("relevance", 0)
        # Weights: factual accuracy and hallucination matter most
        composite = (fc * 0.2 + fa * 0.35 + ha * 0.25 + re * 0.2)
        composite_scores.append(round(composite, 2))

    stats["composite"] = {
        "mean": round(statistics.mean(composite_scores), 2),
        "median": round(statistics.median(composite_scores), 2),
        "min": round(min(composite_scores), 2),
        "max": round(max(composite_scores), 2),
        "stdev": round(statistics.stdev(composite_scores), 2) if len(composite_scores) > 1 else 0,
    }

    # Grade interpretation
    mean_composite = stats["composite"]["mean"]
    if mean_composite >= 9:
        grade = "EXCELLENT — Agent responses are highly accurate and complete"
    elif mean_composite >= 7.5:
        grade = "GOOD — Agent is mostly accurate with minor gaps"
    elif mean_composite >= 6:
        grade = "FAIR — Noticeable accuracy gaps, needs improvement"
    elif mean_composite >= 4:
        grade = "POOR — Significant inaccuracies or missing data"
    else:
        grade = "CRITICAL — Agent responses are unreliable"

    stats["grade"] = grade
    stats["total_judged"] = len(valid)
    stats["total_skipped"] = len(scores) - len(valid)

    return stats


def generate_accuracy_report(results: list[dict], stats: dict) -> str:
    """Generate accuracy_report.md."""
    lines = [
        "# agent Agent — Accuracy Report",
        f"\n**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Queries Judged:** {stats.get('total_judged', 0)}",
        f"**Queries Skipped (no reference):** {stats.get('total_skipped', 0)}",
        f"\n## Overall Grade: {stats.get('grade', 'N/A')}",
    ]

    # Composite
    if "composite" in stats:
        c = stats["composite"]
        lines.append(f"\n### Composite Score (weighted)")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean | **{c['mean']}/10** |")
        lines.append(f"| Median | {c['median']}/10 |")
        lines.append(f"| Min | {c['min']}/10 |")
        lines.append(f"| Max | {c['max']}/10 |")
        lines.append(f"| Std Dev | {c['stdev']} |")
        lines.append(f"\n*Weights: Field Coverage 20%, Factual Accuracy 35%, Hallucination 25%, Relevance 20%*")

    # Per dimension
    dims = ["field_coverage", "factual_accuracy", "hallucination", "relevance"]
    lines.append(f"\n### Per Dimension")
    lines.append("| Dimension | Mean | Median | Mode | Std Dev | Min | Max |")
    lines.append("|-----------|------|--------|------|---------|-----|-----|")
    for dim in dims:
        if dim in stats:
            d = stats[dim]
            lines.append(f"| {dim} | {d['mean']} | {d['median']} | {d['mode']} | {d['stdev']} | {d['min']} | {d['max']} |")

    # Interpretation
    lines.append("\n### Score Interpretation")
    lines.append("| Score Range | Meaning |")
    lines.append("|-------------|---------|")
    lines.append("| 9-10 | Excellent — matches platform exactly |")
    lines.append("| 7.5-9 | Good — minor differences |")
    lines.append("| 6-7.5 | Fair — some data missing or slight inaccuracies |")
    lines.append("| 4-6 | Poor — significant gaps |")
    lines.append("| 0-4 | Critical — unreliable |")

    # Individual scores
    lines.append("\n## Individual Query Accuracy")
    lines.append("| ID | Query | Coverage | Accuracy | Halluc. | Relevance | Composite | Reason |")
    lines.append("|----|-------|----------|----------|---------|-----------|-----------|--------|")
    for r in results:
        s = r.get("accuracy_scores", {})
        if s.get("field_coverage", -1) < 0:
            lines.append(f"| {r['id']} | {r['query'][:50]}... | - | - | - | - | - | {s.get('reason', 'skipped')} |")
            continue
        fc = s.get("field_coverage", 0)
        fa = s.get("factual_accuracy", 0)
        ha = s.get("hallucination", 0)
        re = s.get("relevance", 0)
        composite = round(fc * 0.2 + fa * 0.35 + ha * 0.25 + re * 0.2, 2)
        reason = s.get("reason", "")[:60].replace("|", "/")
        lines.append(f"| {r['id']} | {r['query'][:50]}... | {fc} | {fa} | {ha} | {re} | {composite} | {reason} |")

    return "\n".join(lines)


async def run_accuracy_eval(results_file: Path, ground_truth_file: Path):
    """Main accuracy evaluation runner."""
    # Load bench results
    bench_data = json.loads(results_file.read_text())
    results = bench_data.get("results", [])
    print(f"[ACCURACY] Loaded {len(results)} query results")

    # Load ground truth
    if not ground_truth_file.exists():
        print(f"[ACCURACY] Ground truth file not found: {ground_truth_file}")
        print(f"[ACCURACY] Creating template — fill in 'platform_response' for each query")
        template = [{"id": r["id"], "query": r["query"], "platform_response": ""} for r in results]
        ground_truth_file.write_text(json.dumps(template, indent=2))
        print(f"[ACCURACY] Template saved → {ground_truth_file}")
        print(f"[ACCURACY] Fill in platform responses and re-run.")
        return

    ground_truth = json.loads(ground_truth_file.read_text())
    gt_map = {g["id"]: g.get("platform_response", "") for g in ground_truth}
    print(f"[ACCURACY] Loaded {len(ground_truth)} ground truth entries")

    filled = sum(1 for v in gt_map.values() if v.strip())
    print(f"[ACCURACY] Filled entries: {filled}/{len(gt_map)}")

    if filled == 0:
        print("[ACCURACY] No platform responses filled yet. Fill ground_truth.json and re-run.")
        return

    # Judge each
    scored_results = []
    async with httpx.AsyncClient() as client:
        for i, r in enumerate(results, 1):
            qid = r["id"]
            reference = gt_map.get(qid, "")
            agent_resp = r.get("response", "")

            if not reference.strip():
                scores = {"field_coverage": -1, "factual_accuracy": -1, "hallucination": -1, "relevance": -1, "reason": "No reference"}
            else:
                print(f"[{i:02d}/{len(results)}] Judging {qid}...")
                scores = await judge_response(client, r["query"], agent_resp, reference)
                await asyncio.sleep(0.3)  # rate limit

            r["accuracy_scores"] = scores
            r["platform_response"] = reference
            scored_results.append(r)

    # Compute stats
    all_scores = [r["accuracy_scores"] for r in scored_results]
    stats = compute_accuracy_stats(all_scores)

    # Save
    output = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(scored_results),
            "judged": stats.get("total_judged", 0),
        },
        "stats": stats,
        "results": scored_results,
    }
    ACCURACY_RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n[ACCURACY] Results saved → {ACCURACY_RESULTS_FILE}")

    # Report
    report = generate_accuracy_report(scored_results, stats)
    ACCURACY_REPORT_FILE.write_text(report)
    print(f"[ACCURACY] Report saved → {ACCURACY_REPORT_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    if "composite" in stats:
        c = stats["composite"]
        print(f"  Composite Mean:   {c['mean']}/10")
        print(f"  Composite Median: {c['median']}/10")
        print(f"  Grade:            {stats['grade']}")
    print()
    for dim in ["field_coverage", "factual_accuracy", "hallucination", "relevance"]:
        if dim in stats:
            d = stats[dim]
            print(f"  {dim:20s}  mean={d['mean']:.1f}  median={d['median']:.1f}  stdev={d['stdev']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="agent Agent Accuracy Evaluator")
    parser.add_argument("--results", default=str(RESULTS_FILE), help="Path to bench_results.json")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_FILE), help="Path to ground_truth.json")
    args = parser.parse_args()

    asyncio.run(run_accuracy_eval(Path(args.results), Path(args.ground_truth)))


if __name__ == "__main__":
    main()
