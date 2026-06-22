# agentic_eval

A mini agentic evaluation framework showing how AI labs benchmark tool-using agents: a sandboxed environment, an automated verifier that grades both outcome and process, and adversarial tests that confirm the verifier resists reward hacking.

## Quick start

```bash
pip install -r requirements.txt   # anthropic + pytest

# Run 5 instances with the offline baseline agent (no API key needed)
python -m agentic_eval.run --agent baseline --n 5

# Run with the Anthropic model agent
ANTHROPIC_API_KEY=sk-... python -m agentic_eval.run --agent model --n 3

# Run the test suite (no API key required)
pytest
```

## What it does

The framework evaluates an agent on a **SQL data-analysis task**:

> Given a SQLite database of customers, subscriptions, usage events, and payments, identify the **top 3 customers most at risk of churn** according to an explicit formula, and submit a ranked JSON answer.

The agent interacts with the environment through five tools:

| Tool | Description |
|---|---|
| `list_tables` | List all tables in the database |
| `describe_table(name)` | Show column definitions |
| `run_sql(query)` | Execute a read-only SELECT query |
| `read_file(filename)` | Read a file from the agent's sandbox |
| `submit_answer(answer)` | Write the final answer to `answer.json` |

## Grading

The verifier grades four criteria independently:

| Criterion | What it checks |
|---|---|
| `well_formed_json` | Submission has the correct structure (3 entries, required fields) |
| `correct_ids` | Submitted `customer_id`s match ground truth |
| `correct_order` | Ranking order is correct (highest risk first) |
| `queried_tables` | Trace shows SQL queries on `usage_events`, `payments`, and `subscriptions` |

All four must pass for `verdict.passed = True`. A correct answer submitted without running any queries fails the process criterion — guessing is not rewarded.

## Example output

```
[1/5] churn-seed0 ... PASS
[2/5] churn-seed1 ... PASS
[3/5] churn-seed2 ... PASS
[4/5] churn-seed3 ... PASS
[5/5] churn-seed4 ... PASS

Report saved → report.json

======================================================
  Instances:  5
  Pass rate:  100.0%

  Per-criterion pass rates:
    well_formed_json             100.0%  [####################]
    correct_ids                  100.0%  [####################]
    correct_order                100.0%  [####################]
    queried_tables               100.0%  [####################]
======================================================
```

## File layout

```
agentic_eval/
  __init__.py
  interfaces.py        # Task, Environment, Tool, Verifier, Agent, VerdictReport
  tools.py             # Tool implementations + SQL safety guard
  trace.py             # JSON trace serialisation
  run.py               # CLI runner + human-readable report
  agents/
    baseline.py        # Scripted offline agent (no API key)
    model.py           # Anthropic Messages API agent (tool-use loop)
  tasks/
    churn/
      build_db.py      # Deterministic synthetic database generator
      task.py          # ChurnTask + ChurnEnvironment
      verifier.py      # Ground-truth recomputation + 4-criterion grading
tests/
  test_verifier_robustness.py   # 4 adversarial cases
  test_end_to_end.py            # Multi-seed correctness, CLI, SQL guard, determinism
requirements.txt
README.md
DESIGN.md              # Schema, churn formula, verifier-robustness rationale
```

## CLI flags

```
python -m agentic_eval.run [OPTIONS]

Options:
  --agent {baseline,model}   Agent to use (default: baseline)
  --n N                      Number of task instances (default: 5)
  --seed INT                 Starting seed (default: 0)
  --out PATH                 Report file (default: report.json)
  --traces-dir PATH          Directory for per-instance trace JSON files
```

## Extending

Subclass `Task`, `Environment`, and `Verifier` from `agentic_eval.interfaces` to add a new task. The runner, trace logger, and tool layer are task-agnostic.

See `DESIGN.md` for the verifier-robustness rationale and a full description of the churn formula and data-generation scheme.
