"""Extract the canonical question bank (sheet1) from the agent xlsx."""
import zipfile, json, re
from xml.etree import ElementTree as ET

XLSX = "/Users/fardeenkhatrins/Downloads/Question_Bank_agent.xlsx"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def load():
    z = zipfile.ZipFile(XLSX)
    shared = []
    t = ET.fromstring(z.read("xl/sharedStrings.xml"))
    for si in t.findall(NS + "si"):
        shared.append("".join(x.text or "" for x in si.iter(NS + "t")))

    def rows(sheet):
        out = []
        t = ET.fromstring(z.read(sheet))
        for row in t.iter(NS + "row"):
            cells = []
            for c in row.findall(NS + "c"):
                v = c.find(NS + "v")
                if v is None:
                    cells.append("")
                    continue
                cells.append(shared[int(v.text)] if c.get("t") == "s" else v.text)
            out.append(cells)
        return out

    data = rows("xl/worksheets/sheet1.xml")
    qs = []
    srno = 0
    for r in data[1:]:
        if len(r) < 2:
            continue
        num = (r[0] or "").strip()
        q = (r[1] or "").strip()
        cat = (r[2] or "").strip() if len(r) > 2 else ""
        persona = (r[5] or "").strip() if len(r) > 5 else ""
        if not q:
            continue
        # Take the first non-empty line of the question cell (some cells have alt phrasings)
        first = next((ln.strip() for ln in q.split("\n") if ln.strip()), "")
        if len(first) < 6:
            continue
        srno += 1
        # Prefer the bank's own number if it's a clean integer
        try:
            srno_val = int(float(num))
        except (ValueError, TypeError):
            srno_val = srno
        qs.append({
            "srno": srno_val,
            "question": first,
            "category": cat or "general",
            "persona": persona,
        })
    return qs


if __name__ == "__main__":
    qs = load()
    json.dump(qs, open("/Users/fardeenkhatrins/Documents/internal/agent/agent/bench/question_bank.json", "w"), indent=2, ensure_ascii=False)
    print(f"Extracted {len(qs)} questions")
    for q in qs:
        print(f"  {q['srno']:>3} [{q['category'][:18]:18}] {q['question'][:80]}")
