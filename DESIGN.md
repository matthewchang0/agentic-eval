# Design Notes — `agentic_eval`

## Database Schema

```
customers
  customer_id  INTEGER PK
  name         TEXT
  email        TEXT
  signup_date  TEXT (ISO-8601)

subscriptions
  subscription_id  INTEGER PK
  customer_id      INTEGER FK → customers
  status           TEXT  CHECK IN ('active','cancelled','paused')
  start_date       TEXT (ISO-8601)
  plan             TEXT  CHECK IN ('basic','pro','enterprise')

usage_events
  event_id     INTEGER PK AUTOINCREMENT
  customer_id  INTEGER FK → customers
  event_date   TEXT (ISO-8601)
  event_count  INTEGER DEFAULT 1

payments
  payment_id    INTEGER PK AUTOINCREMENT
  customer_id   INTEGER FK → customers
  payment_date  TEXT (ISO-8601)
  amount        REAL
  status        TEXT  CHECK IN ('success','failed')
```

Each database is generated from a single integer `seed` using Python's
`random.Random(seed)` — no network, no wall-clock, no OS randomness.  The
reference date (the "today" for churn calculations) is fixed at
`2024-03-01` unless overridden, so the date arithmetic is always the same.

## Churn-Risk Formula

Only customers with `subscriptions.status = 'active'` qualify.

```
prior_events     = SUM(event_count) WHERE event_date ∈ [ref−60d, ref−30d)
recent_events    = SUM(event_count) WHERE event_date ∈ [ref−30d, ref)
failed_payments  = COUNT(*)          WHERE status='failed'
                                       AND payment_date ∈ [ref−60d, ref)

usage_decline    = MAX(0, (prior_events − recent_events) / MAX(prior_events, 1))
churn_score      = usage_decline × (1 + failed_payments)
```

Rank by `churn_score DESC`, break ties by `customer_id ASC`.  The answer is
the top 3.

### Why this formula

- **Usage decline** captures disengagement: a customer going quiet is an
  early churn signal.
- **Failed payments** capture payment friction, a separate but compounding
  risk factor.
- Multiplying rather than adding means both signals must be non-zero to
  produce a high score, avoiding false positives (e.g. someone who simply
  never used the product).
- The formula is fully computable with two SQL aggregations + one join,
  so an agent that uses the tools correctly will always find the answer.

### Data-generation tiers

The synthetic generator assigns customers to risk tiers by shuffling IDs
with the RNG — the shuffle is seed-dependent, so the customer IDs in the
top-3 change with every seed:

| Tier        | Count | Status   | Prior events | Decline  | Failed payments |
|-------------|-------|----------|--------------|----------|-----------------|
| HIGH_RISK   | 4     | active   | 30–50        | 70–90 %  | 2–4             |
| MEDIUM_RISK | 4     | active   | 15–30        | 40–60 %  | 1               |
| LOW_RISK    | 6     | active   | 10–25        | −10–25 % | 0               |
| INACTIVE    | 6     | cancelled/paused | 5–20 | 0–60 %   | 0–2             |

Score separation is guaranteed:
- HIGH_RISK minimum: `0.70 × (1+2) = 2.1`
- MEDIUM_RISK maximum: `0.60 × (1+1) = 1.2`

So the top-3 are always three of the four HIGH_RISK customers; which three
and in what order depends on the specific random draws for that seed.

---

## Verifier Robustness

### Threat model

We treat the evaluated agent as adversarial — it may try to:

1. **Guess the answer** without doing real work.
2. **Exfiltrate or overwrite the ground-truth** by reading/writing files.
3. **Inject SQL** to mutate the database and make any ID look correct.
4. **Return a plausible but wrong answer** that exploits lenient grading.

### Defences

| Attack vector | Defence |
|---|---|
| Guess without querying | Process criterion: verifier inspects the trace for SQL queries on `usage_events`, `payments`, and `subscriptions`. A correct answer with no tool calls fails. |
| Read the ground-truth file | Ground truth is never written to a file. The verifier recomputes it from the raw DB at evaluation time, inside the verifier process. |
| Overwrite the database | `run_sql` rejects any non-SELECT statement. A regex guard (`^SELECT\b`) is checked before any DB connection is opened. Additional forbidden-keyword scanning blocks `ATTACH`, `PRAGMA`, etc. |
| Path traversal via `read_file` | The tool resolves the target path and checks that it lies strictly inside `env.working_dir`. Any path escaping the sandbox returns an error dict; the agent loop sees this as a normal (failed) tool result. |
| Mutation via `read_file` | `read_file` is read-only; writing to the DB is only possible via `run_sql`, which is guarded. |
| Hardcode IDs for seed=42 | The tier assignment is shuffled per seed, so any hardcoded ID list is wrong for almost all seeds. The end-to-end tests confirm different seeds yield different ground truths. |
| Inflate score by altering scoring formula | Scoring is done exclusively in the verifier, which runs after teardown and references the original (unmodified) DB. The agent's working dir and the DB live in the same temp directory, but the verifier re-opens the DB path via the `env` object — if the agent corrupted the DB, `PRAGMA integrity_check` would fail and `sqlite3.Error` would bubble up. |

### Why "recompute, don't trust"

Storing expected output anywhere the agent can reach creates an attack
surface.  By keeping ground truth exclusively in the verifier's private
logic, we eliminate the entire class of "read-the-answer-key" attacks.
Even if the agent can read every file in its working directory, there is
nothing useful there until it submits `answer.json`.

### Limitations / future work

- The process criterion checks table *names* in the SQL text; a sufficiently
  creative agent could include the keyword without doing meaningful work
  (e.g. `SELECT 'usage_events'`).  A stronger check would parse the query
  AST and verify that the relevant columns were actually projected.
- The formula is public in the task prompt, which lets a sufficiently
  capable agent implement it in-context without querying the DB.  A harder
  variant would omit the formula and require the agent to infer it from the
  data.
- SQL injection via nested comments or Unicode tricks is not fully guarded;
  for production use, a proper SQL parser should replace the regex.

---

## Architecture decisions

### Why SQLite (not Postgres, DuckDB, etc.)

Zero infrastructure.  The entire task — DB creation, query execution,
verification — runs in a single Python process with no daemons.  SQLite's
`PRAGMA query_only = ON` gives us read-enforcement at the connection level
as a second layer behind the regex guard.

### Why store dates as TEXT

SQLite has no native DATE type.  ISO-8601 strings (`YYYY-MM-DD`) compare
correctly with lexicographic ordering, making range queries with `>=` / `<`
correct and readable.

### Why the agent receives `reference_date` in the prompt

The DB is generated at a fixed reference date, not the actual wall-clock
"today".  Embedding the date in the prompt lets both the model agent and
the scripted baseline agent use the correct date boundaries without any
magic.

### Baseline agent design

The baseline agent is a scripted but *realistic* query sequence:
1. Discover schema (`list_tables`, `describe_table` × 4).
2. Confirm active customers (satisfies `subscriptions` criterion).
3. Spot-check usage trends (satisfies `usage_events` criterion).
4. Spot-check failed payments (satisfies `payments` criterion).
5. Compute full churn scores in a single CTE query.
6. Submit answer.

This mirrors what a competent analyst would do and ensures all four verifier
criteria are satisfied.
