"""Ingest the post-release test workbook into a clean ground-truth JSON.

Source: Book.xlsx, sheet 'Questions_after_release_test_up'
  header row = row 2 (1-indexed); columns: #, Question, SQL Queries, RAW OUTPUT, Agent, Error
Output: release_ground_truth.json — the authoritative per-question record used by
release_eval.py. The RAW OUTPUT column is the raw-DB answer (our reference of truth).

Run on the HOST (needs openpyxl + access to the workbook):
    python apps/agent/bench/release_ingest.py [/path/to/Book.xlsx]
"""
import json
import sys
from pathlib import Path

import openpyxl

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "Book.xlsx"
SHEET = "Questions_after_release_test_up"
OUT = Path(__file__).with_name("release_ground_truth.json")

# 0-indexed columns within the row tuple
C_NUM, C_Q, C_SQL, C_RAW, C_AGENT, C_ERR = 1, 2, 3, 4, 5, 6


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(values_only=True))
    data = rows[2:]  # skip the 2 header rows

    out = []
    for r in data:
        q = r[C_Q] if len(r) > C_Q else None
        if not q or not str(q).strip():
            continue

        def cell(i):
            v = r[i] if len(r) > i else None
            return str(v).strip() if v is not None else ""

        out.append({
            "id": int(r[C_NUM]) if r[C_NUM] not in (None, "") else len(out) + 1,
            "question": str(q).strip(),
            "sql": cell(C_SQL),
            "raw": cell(C_RAW),          # raw-DB output = reference of truth
            "agent_earlier": cell(C_AGENT),
            "noted_error": cell(C_ERR),
        })

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_err = sum(1 for x in out if x["noted_error"])
    print(f"wrote {OUT} | {len(out)} questions | {n_err} with noted errors")


if __name__ == "__main__":
    main()
