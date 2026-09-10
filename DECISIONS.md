# Decisions — agent-claude

Why this agent is built the way it is. Plain English, one entry per decision that would
otherwise look arbitrary to someone reading the code.

This is one of two agents built to be compared. **The only intended difference between them
is where the data comes from.** This one queries the database itself. The other asks the
platform.

---

## 1. The model writes the database queries

Claude gets three read-only tools — list the tables, describe a table, run a query — and
works out how to answer the question itself. It writes real SQL against a real database.

## 2. It uses the same AI model as the other agent

Both run Claude Sonnet, so any cost or quality difference comes from the data path rather
than from one side having a better model.

## 3. Everything except the data step is identical to the other agent

Same code, same steps, same prompts. The swap happens deep inside, at the single point where
data is fetched — so the two agents cannot drift apart in ways nobody notices.

## 4. Safety is enforced in code, never by asking nicely

Every query the model writes is parsed and inspected before the database sees it. Anything
that is not a single, size-limited read is refused with a reason the model can act on.

**Why it is not just an instruction in the prompt:** "we told it to only read" is a request,
not a control. A model under pressure to answer will occasionally write something else.

Three independent layers, because each fails differently:

1. the parser refuses the query outright,
2. the database account itself cannot write, and the session is opened read-only,
3. a row cap and a time limit bound the cost of an expensive-but-legal read.

Two attacks the naive version missed, both now covered by tests: a deletion hidden inside
what looks like a read, and a harmless query followed by a semicolon and something else.

## 5. Arithmetic happens in the database, not in the model's head

Totals, counts, averages and rankings are written as SQL so the **database** computes them.
The model translates the question and phrases the result; it does not add numbers up itself.

**Why this matters more than it sounds.** The row cap trims results to a fixed size. A model
that fetched 505 holdings intending to total them would get 200, sum those, and report a
number **less than half the truth** — with no error and a perfectly plausible answer.
Measured: ₹1.22bn reported against a real ₹2.35bn.

So a result that exactly fills the cap is now flagged as probably incomplete, and the model
is told to rewrite the query to aggregate in SQL. That check fires on a row count, not on
the model behaving well.

## 6. The loop has a hard ceiling

The model can go round the tools as many times as it needs, up to a fixed limit.

**The trade:** looping handles an unfamiliar schema and lets the model correct its own SQL
errors — it has already been observed guessing a wrong column name, reading the error, and
retrying successfully. The price is a variable and higher number of model calls per
question. The ceiling stops a confused turn from billing without bound.

## 7. It hands back the same shape of data as the other agent

Rows come back in exactly the format the platform path produces.

**Why:** everything downstream — understanding the question, writing the answer, checking
it, drawing charts — then needs no changes at all. Without this the two agents would differ
in dozens of places instead of one, and the comparison would be meaningless.

## 8. No platform, and no company name anywhere

This agent uses none of the company's running services for data. The repository carries
neither the product name nor the company name.

**What that cost the experiment, stated honestly:** the original plan was for this agent to
replace only the data lookup and keep using the platform for its skills list and its map of
answerable questions. Dropping those too means the two agents now differ in three ways, not
one — so if this one is cheaper or dearer, we can no longer say confidently that the data
path is why. Independence was judged more important than attribution.

## 9. Its working is visible, and that has already paid off

Because the SQL is readable, an answer can be checked against the database by hand.

That is how a real error was caught: on one question the other agent reported a total about
six times too high, and this agent's query — which we could read and verify — showed the
correct figure. Cost per turn is the easy number; **cost per correct answer** is the one
that matters.
