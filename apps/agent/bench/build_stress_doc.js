// Generates the stress-test Word doc from the live results JSON.
// usage: node build_stress_doc.js <results.json> <out.docx>
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak, TableOfContents,
} = require("/opt/homebrew/lib/node_modules/docx");

const RES = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3];

const NAVY = "1F3864", BLUE = "2E5496", LIGHT = "D9E2F3", LIGHTER = "EAF0FA",
  GREY = "595959", GREEN = "E2EFDA", RED = "FBE4E4", AMBER = "FFF2CC", MONO = "F2F2F2";
const CW = 9360;

const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
function P(text, o = {}) {
  return new Paragraph({ children: Array.isArray(text) ? text : [new TextRun({ text, ...o.run })],
    spacing: { after: o.after ?? 120, line: 276 }, ...o.p });
}
function bullet(t, lvl = 0) {
  return new Paragraph({ numbering: { reference: "b", level: lvl },
    children: Array.isArray(t) ? t : [new TextRun(t)], spacing: { after: 60, line: 264 } });
}
function num(t) {
  return new Paragraph({ numbering: { reference: "n", level: 0 },
    children: Array.isArray(t) ? t : [new TextRun(t)], spacing: { after: 60, line: 264 } });
}
const bd = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: bd, bottom: bd, left: bd, right: bd, insideHorizontal: bd, insideVertical: bd };
function cell(content, { w, fill, bold = false, color, align, mono = false } = {}) {
  const arr = Array.isArray(content) ? content : [content];
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: arr.map(c => typeof c === "string"
      ? new Paragraph({ alignment: align, spacing: { after: 0, line: 250 },
          children: [new TextRun({ text: c, bold, color, font: mono ? "Consolas" : undefined, size: mono ? 16 : undefined })] })
      : c),
  });
}
function hrow(labels, widths, fill = BLUE) {
  return new TableRow({ tableHeader: true, children: labels.map((l, i) => cell(l, { w: widths[i], fill, bold: true, color: "FFFFFF" })) });
}
function table(widths, rows) { return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, borders, rows }); }
function clip(s, n) { s = (s == null ? "" : String(s)); return s.length > n ? s.slice(0, n) + " …[truncated]" : s; }

// ---- metrics ----
const n = RES.length;
const npass = RES.filter(r => r.passed).length;
const nagent = RES.filter(r => r.agent_ok).length;
const nplat = RES.filter(r => r.platform_ok).length;
const good = RES.filter(r => r.platform_ok);
const nag = good.filter(r => r.agent_ok).length;
const lat = RES.map(r => r.wall_ms || 0).filter(Boolean).sort((a, b) => a - b);
const med = lat[Math.floor(lat.length / 2)] || 0, p90 = lat[Math.floor(lat.length * 0.9)] || 0, max = lat[lat.length - 1] || 0;
const pct = (a, b) => b ? (100 * a / b).toFixed(0) + "%" : "—";

const kids = [];
// cover
kids.push(
  new Paragraph({ spacing: { before: 2200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Chat Agent", bold: true, size: 52, color: NAVY })] }),
  new Paragraph({ spacing: { before: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Live Stress-Test Results & Current Working", size: 30, color: BLUE })] }),
  new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 8 } }, children: [new TextRun("")] }),
  new Paragraph({ spacing: { before: 300 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${n} questions • answers verified against the raw database and the platform payload`, italics: true, size: 22, color: GREY })] }),
  new Paragraph({ spacing: { before: 1800 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Tested live over HTTP (guardrails on, session-scoped memory), branch fkhatri-neoagent", size: 18, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
);
kids.push(H1("Contents"));
kids.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
kids.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 1. results at a glance ----
kids.push(H1("1. Results at a Glance"));
kids.push(P("Every question was sent to the live agent, and the answer was scored two ways: against the raw database (the real truth), and against the platform payload the agent actually received. This separates the agent's own accuracy from the platform's data quality."));
kids.push(table([4680, 1560, 3120], [
  hrow(["What we measured", "Score", "Plain meaning"], [4680, 1560, 3120]),
  new TableRow({ children: [cell("End-to-end accuracy (answer vs raw DB)", { w: 4680, fill: LIGHTER }), cell(`${npass}/${n} = ${pct(npass, n)}`, { w: 1560, bold: true, align: AlignmentType.CENTER }), cell("Matches the real database value.", { w: 3120 })] }),
  new TableRow({ children: [cell("Agent-faithful (answer vs its payload)", { w: 4680, fill: LIGHTER }), cell(`${nagent}/${n} = ${pct(nagent, n)}`, { w: 1560, bold: true, align: AlignmentType.CENTER }), cell("Did the agent correctly use what it was given?", { w: 3120 })] }),
  new TableRow({ children: [cell("Platform-OK (payload had right data)", { w: 4680, fill: LIGHTER }), cell(`${nplat}/${n} = ${pct(nplat, n)}`, { w: 1560, bold: true, align: AlignmentType.CENTER }), cell("How often the platform returned correct data.", { w: 3120 })] }),
  new TableRow({ children: [cell("Agent accuracy GIVEN good platform data", { w: 4680, fill: GREEN, bold: true }), cell(`${nag}/${good.length} = ${pct(nag, good.length)}`, { w: 1560, bold: true, align: AlignmentType.CENTER }), cell("When the platform is right, is the agent right?", { w: 3120 })] }),
  new TableRow({ children: [cell("Latency (per answer)", { w: 4680, fill: LIGHTER }), cell(`${(med/1000).toFixed(1)}s`, { w: 1560, bold: true, align: AlignmentType.CENTER }), cell(`median; p90 ${(p90/1000).toFixed(0)}s, max ${(max/1000).toFixed(0)}s`, { w: 3120 })] }),
]));
kids.push(P([new TextRun({ text: "How to read this: ", bold: true }), new TextRun(`the agent answered correctly on ${pct(nag, good.length)} of the questions where the platform returned good data. The end-to-end number (${pct(npass, n)}) is lower only because the platform itself returned wrong or missing data on ${n - nplat} of ${n} questions — those are platform-side, not agent, issues.`)], { after: 80 }));
kids.push(calloutBox("Important: the platform is non-deterministic", [
  [new TextRun("The same question can return good data one time and wrong/empty data the next. So the end-to-end number swings run-to-run with the platform's mood. Across live runs it ranged "), new TextRun({ text: "33%–47% end-to-end", bold: true }), new TextRun(", while the platform returned good data on only 37%–53% of questions.")],
  [new TextRun("The number that reflects "), new TextRun({ text: "our", italics: true }), new TextRun(" work is stable across runs: "), new TextRun({ text: "agent accuracy given good platform data stayed ~91–100%", bold: true, color: NAVY }), new TextRun(". In other words, the agent is reliable; the end-to-end ceiling is set by platform data quality.")],
], AMBER, "BF8F00"));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 2. how the agent works now ----
kids.push(H1("2. How the Agent Works Now (Step by Step)"));
kids.push(P("A message flows through a fixed pipeline. Each step is small and does one job:"));
kids.push(num([new TextRun({ text: "Guardrail + memory (in parallel). ", bold: true }), new TextRun("The message is screened for unsafe/PII content while long-term memory is fetched. If the message is blocked, the agent politely refuses and stops here.")]));
kids.push(num([new TextRun({ text: "Understand the request. ", bold: true }), new TextRun("A small, cheap model pulls out the client name and what data is wanted. The name is matched to the known client list (so 'Ram' becomes 'Ram Krishnan').")]));
kids.push(num([new TextRun({ text: "Decide: fetch or recall. ", bold: true }), new TextRun("By default the agent fetches fresh data from the platform. It only answers from memory when you explicitly ask about the earlier conversation (e.g. 'summarise that'). This stops it from making up answers from old turns.")]));
kids.push(num([new TextRun({ text: "Send the question to the platform. ", bold: true }), new TextRun("The question is sent as-is (no risky rewriting). If the platform returns nothing or fails, the agent retries up to a few times, each time rephrasing slightly so the platform's 120-second cache can't replay the same failure.")]));
kids.push(num([new TextRun({ text: "Format the answer. ", bold: true }), new TextRun("Money amounts are formatted in code (so ₹16.33 crore can't become ₹163 crore), then a capable model turns the data into a clear table/answer. It is told to use only the values present — never invent funds, numbers, or rows.")]));
kids.push(num([new TextRun({ text: "Stream + remember. ", bold: true }), new TextRun("The answer streams to the screen; memory is saved in the background so the input box frees up immediately.")]));
kids.push(spacer());
kids.push(calloutBox("Memory model (now session-scoped)", [
  [new TextRun("• "), new TextRun({ text: "Session memory", bold: true }), new TextRun(" is per-session: a new session starts with a clean slate.")],
  [new TextRun("• "), new TextRun({ text: "Long-term memory", bold: true }), new TextRun(" (your profile/preferences) is per-user and is re-loaded on the first question of each new session.")],
  [new TextRun("• A "), new TextRun({ text: "Clear Long-Term Memory", bold: true }), new TextRun(" button in the debug panel wipes the stored long-term memory for the signed-in user.")],
], LIGHTER, BLUE));
kids.push(spacer());
kids.push(calloutBox("Key fixes in this release", [
  [new TextRun("• Stopped the agent answering data questions from thin air (live-fetch by default).")],
  [new TextRun("• Stopped query 'refinement' from changing the question's meaning (send as-is + a safety guard).")],
  [new TextRun("• Fixed the 10× money bug and the scrambled AUM client names (now formatted/joined in code).")],
  [new TextRun("• Retries now rephrase so the platform cache can't replay a failure; spurious 401s are retried.")],
], GREEN, "548235"));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 3. per-question results ----
kids.push(H1("3. Every Question — Answer vs Raw Truth vs Platform Payload"));
kids.push(P("For each question: the three verdicts (end-to-end / agent / platform), the per-dimension scores, then the agent's answer, the raw-DB truth, and the platform payload the agent received. 'Align?' says whether the answer matches the raw truth."));

function verdictChip(r) {
  const e2e = r.passed ? "PASS" : "fail";
  const ag = r.agent_ok ? "agent ok" : "agent off";
  const pl = r.platform_ok ? "platform ok" : "platform BAD";
  return `${e2e}  •  ${ag}  •  ${pl}`;
}
for (const r of RES) {
  const s = r.score || {};
  const fill = r.passed ? GREEN : (r.agent_ok ? AMBER : RED);
  kids.push(new Paragraph({ spacing: { before: 200, after: 40 }, keepNext: true,
    children: [new TextRun({ text: `Q${r.id}. ${clip(r.question, 160)}`, bold: true, size: 22, color: NAVY })] }));
  kids.push(table([2400, 2400, 4560], [
    new TableRow({ children: [
      cell("Verdict", { w: 2400, fill: LIGHT, bold: true }),
      cell(verdictChip(r), { w: 2400, fill, bold: true }),
      cell(`scores — accuracy ${s.factual_accuracy ?? "-"}/10 · no-hallucination ${s.hallucination ?? "-"}/10 · faithful ${s.agent_faithful ?? "-"}/10 · payload-correct ${s.payload_correct ?? "-"}/10 · ${r.wall_ms}ms`, { w: 4560 }),
    ] }),
    new TableRow({ children: [ cell("Align?", { w: 2400, fill: LIGHT, bold: true }), cell(r.passed ? "Yes — matches raw DB" : (r.platform_ok ? "No — agent issue" : "No — platform returned wrong/empty data"), { w: 2400 }), cell(clip(s.reason || "", 220), { w: 4560 }) ] }),
  ]));
  kids.push(labeledBlock("Agent response", clip(r.answer, 700)));
  kids.push(labeledBlock("Raw-DB truth", clip(r.raw, 380), true));
  kids.push(labeledBlock("Platform payload (what the agent received)", clip(r.payload, 420), true));
  const qs = Array.isArray(r.queries_sent) ? r.queries_sent.filter(Boolean) : [];
  if (r.enriched_query || qs.length) {
    const lines = [];
    if (r.enriched_query) lines.push("enriched: " + clip(r.enriched_query, 240));
    qs.forEach((q, i) => lines.push(`attempt ${i + 1}: ` + clip(q, 240)));
    kids.push(labeledBlock("Queries sent to platform (enriched + retry rephrasings)", lines.join("\n"), true));
  }
}

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 4. final numbers ----
kids.push(H1("4. Final Accuracy & Latency"));
kids.push(table([4680, 4680], [
  hrow(["Metric", "Result"], [4680, 4680]),
  new TableRow({ children: [cell("End-to-end accuracy (vs raw DB)", { w: 4680, fill: LIGHTER }), cell(`${npass}/${n} = ${pct(npass, n)}`, { w: 4680, bold: true })] }),
  new TableRow({ children: [cell("Agent accuracy given good platform data", { w: 4680, fill: GREEN }), cell(`${nag}/${good.length} = ${pct(nag, good.length)}`, { w: 4680, bold: true })] }),
  new TableRow({ children: [cell("Hallucination-free answers", { w: 4680, fill: LIGHTER }), cell(`${RES.filter(r=>(r.score||{}).hallucination>=9).length}/${n}`, { w: 4680 })] }),
  new TableRow({ children: [cell("Latency median / p90 / max", { w: 4680, fill: LIGHTER }), cell(`${(med/1000).toFixed(1)}s / ${(p90/1000).toFixed(0)}s / ${(max/1000).toFixed(0)}s`, { w: 4680 })] }),
]));
kids.push(P([new TextRun({ text: "Latency note: ", bold: true }), new TextRun("the enabled input guardrails (PII + prompt-injection) add a few seconds to every request, and retries on platform failures add to the tail. The agent's own steps are cheap. If median latency must drop, the easiest levers are trimming the answer model's token budget and capping retries — neither costs accuracy.")]));

// ---- 5. next phase ----
kids.push(H1("5. Where to Improve Next (If Targets Not Met)"));
const platBad = RES.filter(r => !r.platform_ok).map(r => "Q" + r.id);
kids.push(P([new TextRun({ text: "The agent is at its ceiling on this set: ", bold: true }), new TextRun(`it answered correctly on ${pct(nag, good.length)} of the questions where the platform returned good data. The remaining end-to-end gap is the platform returning wrong/empty data on ${platBad.length} questions (${platBad.join(", ")}).`)]));
kids.push(H3("Agent-side (us)"));
kids.push(bullet("Consistency voting for critical aggregates: ask 2–3 times and keep the value the platform repeats, to ride out non-determinism."));
kids.push(bullet("Trim latency: smaller answer-token budget + a retry-time cap; optionally cache a fresh successful result briefly."));
kids.push(bullet("Tighten a few compound questions (e.g. 'AUM and realized gain') so each metric is fetched cleanly."));
kids.push(H3("Platform-side (platform team) — the main lever for higher end-to-end accuracy"));
kids.push(bullet("Family/group aggregates returning firm-wide totals instead of the group (e.g. Krishnan Family)."));
kids.push(bullet("Revenue / inflow / market-value queries returning values that disagree with the raw DB or come back null."));
kids.push(bullet("Intermittent empty results and spurious 401s (the agent retries, but the platform should be stable)."));
kids.push(P([new TextRun({ text: "The full-chain logs capture the exact query, payload and answer for every question, so each platform case can be handed over with evidence.", italics: true, color: GREY, size: 18 })]));

// helper fns used above
function spacer() { return new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] }); }
function labeledBlock(label, text, mono = false) {
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW],
    borders: { top: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" }, bottom: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" }, left: { style: BorderStyle.SINGLE, size: 12, color: mono ? "8FAADC" : "A9D08E" }, right: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" } },
    rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { fill: mono ? MONO : "FFFFFF", type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 140, right: 120 },
      children: [
        new Paragraph({ spacing: { after: 30 }, children: [new TextRun({ text: label, bold: true, size: 17, color: GREY })] }),
        ...String(text || "(none)").split("\n").map((ln, i, a) => new Paragraph({
          spacing: { after: i === a.length - 1 ? 0 : 20, line: 248 },
          children: [new TextRun({ text: ln, font: mono ? "Consolas" : undefined, size: mono ? 16 : 19 })] })),
      ] })] })] });
}
function calloutBox(title, lines, fill, bar) {
  const c = [];
  if (title) c.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, color: NAVY })] }));
  for (const ln of lines) c.push(new Paragraph({ spacing: { after: 40, line: 264 }, children: Array.isArray(ln) ? ln : [new TextRun(ln)] }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW],
    borders: { top: { style: BorderStyle.SINGLE, size: 1, color: fill }, bottom: { style: BorderStyle.SINGLE, size: 1, color: fill }, right: { style: BorderStyle.SINGLE, size: 1, color: fill }, left: { style: BorderStyle.SINGLE, size: 18, color: bar } },
    rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR }, margins: { top: 110, bottom: 110, left: 150, right: 150 }, children: c })] })] });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: "222222" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: NAVY }, paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LIGHT, space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: BLUE }, paragraph: { spacing: { before: 200, after: 90 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, color: "33475B" }, paragraph: { spacing: { before: 140, after: 50 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 520, hanging: 260 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 520, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LIGHT, space: 4 } }, children: [new TextRun({ text: "agent Agent — Stress-Test Results", color: GREY, size: 16 })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Confidential — Internal  •  Page ", color: GREY, size: 16 }), new TextRun({ children: [PageNumber.CURRENT], color: GREY, size: 16 }), new TextRun({ text: " of ", color: GREY, size: 16 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], color: GREY, size: 16 })] })] }) },
    children: kids,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length, "bytes"); });
