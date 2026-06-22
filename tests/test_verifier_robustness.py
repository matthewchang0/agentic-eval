"""
Adversarial verifier-robustness tests.

Four cases:
  1. no_tool_use   — correct answer submitted without querying the DB
                     → must fail the queried_tables criterion
  2. wrong_ids     — obviously wrong customer IDs submitted
                     → must fail correct_ids (and correct_order)
  3. path_traversal — agent tries to read a file outside the sandbox
                      → access must be blocked and process criterion must fail
  4. correct_agent  — baseline agent (correct answer + all queries)
                      → must pass all four criteria
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from agentic_eval.interfaces import TraceStep
from agentic_eval.agents.baseline import BaselineAgent
from agentic_eval.tasks.churn.task import ChurnTask
from agentic_eval.tasks.churn.verifier import ChurnVerifier
from agentic_eval.tools import ReadFileTool, SubmitAnswerTool, default_tools

SEED = 42
REF_DATE = date(2024, 3, 1)


@pytest.fixture(scope="module")
def task_and_ground_truth():
    """Build a shared task + environment for all adversarial tests."""
    task = ChurnTask(seed=SEED, reference_date=REF_DATE)
    env = task.build_env()
    verifier = ChurnVerifier(REF_DATE)
    ground_truth = verifier.compute_ground_truth(env.db_path)
    yield task, env, verifier, ground_truth
    env.teardown()


# ---------------------------------------------------------------------------
# Case 1: correct answer, zero tool calls → process criterion must fail
# ---------------------------------------------------------------------------

def test_no_tool_use_fails_process(task_and_ground_truth):
    """
    An agent that somehow knows the answer but never queries any table
    must fail the queried_tables criterion.
    """
    task, env, verifier, ground_truth = task_and_ground_truth
    correct_ids = [r["customer_id"] for r in ground_truth]

    # Build the correct answer payload
    correct_answer = {
        "top_churn_customers": [
            {"customer_id": cid, "justification": "known externally"}
            for cid in correct_ids
        ]
    }

    # Write it directly, no tool calls recorded
    answer_path = env.working_dir / "answer.json"
    answer_path.write_text(json.dumps(correct_answer))

    # Empty trace — no tool calls at all
    trace: list[TraceStep] = []

    verdict = verifier.evaluate(task, env, trace)

    assert not verdict.passed, "Should fail when process check fails"

    criterion_map = {c.name: c for c in verdict.criteria}
    assert criterion_map["well_formed_json"].passed, "Submission should parse fine"
    assert criterion_map["correct_ids"].passed, "IDs are correct"
    assert criterion_map["correct_order"].passed, "Order is correct"
    assert not criterion_map["queried_tables"].passed, (
        "Must fail: no SQL queries were issued"
    )


# ---------------------------------------------------------------------------
# Case 2: hardcoded / guessed IDs → id and order criteria must fail
# ---------------------------------------------------------------------------

def test_wrong_ids_fail(task_and_ground_truth):
    """
    Submitting plausible but wrong customer IDs must fail the id criterion.
    We use IDs that are guaranteed not to be the top-3 for this seed.
    """
    task, env, verifier, ground_truth = task_and_ground_truth
    correct_ids = {r["customer_id"] for r in ground_truth}

    # Pick IDs that definitely are NOT in the correct set
    wrong_ids = [cid for cid in range(1, 21) if cid not in correct_ids][:3]
    assert len(wrong_ids) == 3, "Need 3 wrong IDs — data invariant broken"

    wrong_answer = {
        "top_churn_customers": [
            {"customer_id": cid, "justification": "guessed"}
            for cid in wrong_ids
        ]
    }
    (env.working_dir / "answer.json").write_text(json.dumps(wrong_answer))

    # Provide process-valid trace (queries all tables) so only outcome fails
    tools = default_tools()
    fake_sql_calls = [
        TraceStep(0, "tool_call", {"tool": "run_sql", "arguments": {"query": "SELECT * FROM subscriptions"}}),
        TraceStep(0, "tool_result", {"tool": "run_sql", "result": {}}),
        TraceStep(1, "tool_call", {"tool": "run_sql", "arguments": {"query": "SELECT * FROM usage_events"}}),
        TraceStep(1, "tool_result", {"tool": "run_sql", "result": {}}),
        TraceStep(2, "tool_call", {"tool": "run_sql", "arguments": {"query": "SELECT * FROM payments"}}),
        TraceStep(2, "tool_result", {"tool": "run_sql", "result": {}}),
    ]

    verdict = verifier.evaluate(task, env, fake_sql_calls)

    assert not verdict.passed
    criterion_map = {c.name: c for c in verdict.criteria}
    assert not criterion_map["correct_ids"].passed, "Wrong IDs must fail"
    assert not criterion_map["correct_order"].passed, "Wrong order must fail"
    assert criterion_map["queried_tables"].passed, "Process criterion should pass"


# ---------------------------------------------------------------------------
# Case 3: path-traversal attempt → access blocked, process fails
# ---------------------------------------------------------------------------

def test_path_traversal_blocked(task_and_ground_truth):
    """
    An agent that tries to read a file outside the sandbox must be blocked.
    The tool must return an error (not raise), and the verifier must reject
    because no legitimate SQL queries were made.
    """
    task, env, verifier, ground_truth = task_and_ground_truth
    read_tool = ReadFileTool()

    # Attempt to escape the sandbox
    traversal_attempts = [
        "../verifier.py",
        "../../agentic_eval/tasks/churn/verifier.py",
        "/etc/passwd",
    ]
    for path in traversal_attempts:
        result = read_tool.execute(env, filename=path)
        assert "error" in result, f"Expected error for path {path!r}, got {result}"
        assert "outside" in result["error"].lower() or "denied" in result["error"].lower(), (
            f"Error message should mention denial: {result['error']}"
        )

    # Even if the agent somehow submitted a correct-looking answer,
    # failing the process check means it won't fully pass.
    correct_ids = [r["customer_id"] for r in ground_truth]
    (env.working_dir / "answer.json").write_text(
        json.dumps({
            "top_churn_customers": [
                {"customer_id": cid, "justification": "obtained via traversal"}
                for cid in correct_ids
            ]
        })
    )

    # Trace has a traversal tool call but no SQL queries on required tables
    trace: list[TraceStep] = [
        TraceStep(0, "tool_call", {"tool": "read_file", "arguments": {"filename": "../verifier.py"}}),
        TraceStep(0, "tool_result", {"tool": "read_file", "result": {"error": "Access denied"}}),
    ]

    verdict = verifier.evaluate(task, env, trace)
    criterion_map = {c.name: c for c in verdict.criteria}
    assert not criterion_map["queried_tables"].passed, (
        "No SQL queries on required tables — process must fail"
    )
    assert not verdict.passed


# ---------------------------------------------------------------------------
# Case 4: correct agent → all criteria pass
# ---------------------------------------------------------------------------

def test_baseline_agent_passes_all_criteria():
    """The baseline agent must pass every verifier criterion on a fresh env."""
    task = ChurnTask(seed=SEED, reference_date=REF_DATE)
    env = task.build_env()
    try:
        tools = default_tools()
        agent = BaselineAgent()
        trace = agent.run(task, tools, env, max_steps=30)

        verifier = ChurnVerifier(REF_DATE)
        verdict = verifier.evaluate(task, env, trace)

        assert verdict.passed, (
            f"Baseline agent should pass all criteria. Verdict:\n"
            + "\n".join(f"  {c.name}: {c.passed} — {c.detail}" for c in verdict.criteria)
        )
        assert verdict.score == 1.0
        for criterion in verdict.criteria:
            assert criterion.passed, f"Criterion {criterion.name!r} failed: {criterion.detail}"
    finally:
        env.teardown()
