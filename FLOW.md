# How agent-claude works

A walkthrough of one question, from typing it to reading the answer.

**In one line:** the agent writes its own database queries, runs them read-only, and turns
the rows into a readable answer.

---

## The pieces

| | |
|---|---|
| **Chat page** | served by the agent itself, at `http://localhost:8001` |
| **The agent** | decides what to do, writes the SQL, writes the answer |
| **SQL guard** | inspects every query before the database sees it |
| **Database** | Postgres, read-only, holds the wealth data |
| **Memory** | remembers facts about the user across conversations |
| **Recorder** | writes down every step and every model call, for the dashboard |

---

## One question, start to finish

```
You type a question
        │
        ▼
   ┌─────────────────────── the agent ────────────────────────┐
   │                                                          │
   │   These two run AT THE SAME TIME:                        │
   │                                                          │
   │   ┌── safety + memory ──┐   ┌── understand + fetch ───┐  │
   │   │ • is this question  │   │ • what are they asking? │  │
   │   │   allowed?          │   │ • which client? which   │  │
   │   │ • what do we know   │   │   metric? which period? │  │
   │   │   about this user?  │   │ • query the database ───┼──┼──► the loop
   │   └─────────────────────┘   └─────────────────────────┘  │      below
   │                                                          │      │
   │   Safety check failed? → refuse, throw the data away      │   rows back
   │   Passed? ─────────────────────────────────────────────► │ ◄────┘
   │                                                          │
   │   Write the answer, streaming it word by word            │
   │   Check the answer is safe to show                       │
   └──────────────────────────────────────────────────────────┘
        │
        ▼
You see the answer appear
```

### Why safety and data run together

The agent does not wait to be told the question is safe before it starts looking things up.
It starts both at once and throws the lookup away if the safety check fails — wasting a
lookup on blocked questions to save real time on the ones that are not.

---

## The data step: the loop

This is the only part that differs from the other agent, and it is where most of the work
happens.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
    Claude looks at the question                       │
        │                                              │
        ├── "what tables exist?" ──────► list tables ──┤
        ├── "what's in this one?" ─────► describe ─────┤
        └── "run this SELECT" ─────┐                   │
                                   ▼                   │
                            ┌─── SQL guard ───┐        │
                            │ single read?    │        │
                            │ no writes?      │        │
                            │ row cap added   │        │
                            └────────┬────────┘        │
                              refused │ allowed        │
                                 │    ▼                │
                    reason ◄─────┘  database (read-only)
                    goes back            │             │
                    to Claude ───────────┴── rows ─────┘
                                                       │
                    enough to answer? ─────────────────┘
                                  │ yes
                                  ▼
                          hand the rows back
```

Claude keeps going round until it has what it needs, up to a fixed limit.

**A real example.** Asked for the client with the highest portfolio value, it wrote a query
joining on `client_id`, got a real error back saying that column does not exist, rewrote it
using `id`, and got the answer. Three round trips, self-corrected, no human involved.

---

## The three safety layers

Every query passes through all three. Each fails in a different way, which is the point.

| Layer | What it stops | If it has a bug |
|---|---|---|
| **The parser** | Anything that is not one bounded read — writes, deletions, table drops, two statements stapled together | A valid query is wrongly refused. Costs one retry. |
| **The account** | The database login itself cannot write, and the session is opened read-only | Postgres refuses the write regardless of what the code did |
| **The limits** | A row cap and a time limit | A huge query is cut short rather than running away |

The checked query is also **rebuilt from what the parser understood**, rather than patched as
text — so what runs is exactly what was inspected, with no room for a quoting trick to mean
two different things.

---

## Why the maths is done by the database

Claude writes `SUM(...)` and `ORDER BY ...` and lets Postgres compute. It does not fetch rows
and add them up itself.

That is not a style preference. The row cap trims results to a fixed size, so a model
totalling fetched rows would silently work from a partial set:

```
Database does the maths:     ₹2,348,178,644   ← correct
Model adds up fetched rows:  ₹1,221,150,021   ← 48% short, no error, looks fine
```

A result that exactly fills the cap is now flagged as probably incomplete, and the model is
told to aggregate in SQL instead.

---

## What you actually see on screen

```
parallel prep       Running input safety scan + loading memory
parallel prep       Safety check passed
intent enrichment   Intent classified as "fetch_data"
data fetch          Querying the database directly (model-written SQL)
data fetch          Asking the database: "Show the AUM breakdown by client tier"
data fetch          1/1 database query succeeded
generate response   Composing answer from 1 data source(s)

## AUM Breakdown by Client Tier
| Tier | Clients | Total AUM | % of Total |
...
```

---

## What gets recorded

Every turn produces a record of the whole turn, one record per model call with tokens in and
out, and a one-line summary row. All tagged `agent-claude`.

**Expect a higher call count than the other agent** — typically 5–6 model calls per question
against its 2–3, because the exploring and the SQL-writing happen here rather than on a
platform. That is the cost of doing the work in the open, and it is the main thing the
comparison measures.
