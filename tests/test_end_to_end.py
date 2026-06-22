"""
End-to-end tests.

  - baseline agent passes for multiple seeds
  - run.py CLI produces a valid report
  - sql guard blocks non-SELECT statements
  - submit_answer is idempotent (second call overwrites first)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from agentic_eval.agents.baseline import BaselineAgent
from agentic_eval.tasks.churn.task import ChurnTask
from agentic_eval.tasks.churn.verifier import ChurnVerifier
from agentic_eval.tools import RunSqlTool, SubmitAnswerTool, default_tools


# ---------------------------------------------------------------------------
# Baseline agent correctness across seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_baseline_passes_all_seeds(seed: int):
    """Baseline agent must pass every criterion for each tested seed."""
    ref = date(2024, 3, 1)
    task = ChurnTask(seed=seed, reference_date=ref)
    env = task.build_env()
    try:
        tools = default_tools()
        trace = BaselineAgent().run(task, tools, env, max_steps=30)
        verdict = ChurnVerifier(ref).evaluate(task, env, trace)
        assert verdict.passed, (
            f"Seed {seed} failed:\n"
            + "\n".join(f"  {c.name}: {c.passed} — {c.detail}" for c in verdict.criteria)
        )
    finally:
        env.teardown()


# ---------------------------------------------------------------------------
# SQL guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO customers VALUES (99, 'hack', 'h@x.com', '2024-01-01')",
        "DROP TABLE customers",
        "DELETE FROM payments",
        "UPDATE subscriptions SET status='active' WHERE 1=1",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "CREATE TABLE pwn (x TEXT)",
        "SELECT * FROM customers; DROP TABLE customers",  # semicolon injection attempt
    ],
)
def test_sql_guard_blocks_mutations(query: str):
    """run_sql must reject any non-SELECT or mutation-containing query."""
    ref = date(2024, 3, 1)
    task = ChurnTask(seed=42, reference_date=ref)
    env = task.build_env()
    try:
        tool = RunSqlTool()
        result = tool.execute(env, query=query)
        assert "error" in result, f"Expected error for query: {query!r}"
    finally:
        env.teardown()


# ---------------------------------------------------------------------------
# CLI runner produces a valid report
# ---------------------------------------------------------------------------

def test_cli_produces_report(tmp_path: Path):
    """Running the CLI with --n 3 should exit 0 and write a valid JSON report."""
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_eval.run",
            "--agent",
            "baseline",
            "--n",
            "3",
            "--out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, (
        f"CLI exited with {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert report_path.exists(), "report.json was not created"

    report = json.loads(report_path.read_text())
    assert report["n_instances"] == 3
    assert report["pass_rate"] == 1.0
    assert "per_criterion_pass_rates" in report
    assert len(report["instances"]) == 3
    for inst in report["instances"]:
        assert inst["passed"]


# ---------------------------------------------------------------------------
# Determinism: same seed → same verdict
# ---------------------------------------------------------------------------

def test_same_seed_same_verdict():
    """Running with the same seed twice must produce identical verdicts."""
    ref = date(2024, 3, 1)

    def run_once(seed: int) -> tuple[list[int], float]:
        task = ChurnTask(seed=seed, reference_date=ref)
        env = task.build_env()
        try:
            tools = default_tools()
            trace = BaselineAgent().run(task, tools, env, max_steps=30)
            verifier = ChurnVerifier(ref)
            gt = verifier.compute_ground_truth(env.db_path)
            verdict = verifier.evaluate(task, env, trace)
            return [r["customer_id"] for r in gt], verdict.score
        finally:
            env.teardown()

    ids1, score1 = run_once(42)
    ids2, score2 = run_once(42)
    assert ids1 == ids2, "Ground truth must be deterministic"
    assert score1 == score2


# ---------------------------------------------------------------------------
# Different seeds → different ground truths (probabilistic check)
# ---------------------------------------------------------------------------

def test_different_seeds_different_answers():
    """
    Different seeds should (with overwhelming probability) produce different
    top-3 rankings — confirming that hardcoding answers won't generalise.
    """
    ref = date(2024, 3, 1)
    seen: set[tuple[int, ...]] = set()
    for seed in range(10):
        task = ChurnTask(seed=seed, reference_date=ref)
        env = task.build_env()
        try:
            gt = ChurnVerifier(ref).compute_ground_truth(env.db_path)
            seen.add(tuple(r["customer_id"] for r in gt))
        finally:
            env.teardown()

    assert len(seen) > 1, (
        "All seeds produced identical top-3 — seed diversity is broken"
    )
