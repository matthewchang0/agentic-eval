"""
Churn-task verifier.

Ground truth is always recomputed from the raw database — never read from any
file the agent could have touched.  Four criteria are graded:

  (a) well_formed_json   — submission is correctly structured JSON
  (b) correct_ids        — submitted customer_ids match the ground-truth set
  (c) correct_order      — ranking order matches ground truth
  (d) queried_tables     — the trace shows the agent queried usage_events,
                           payments, AND subscriptions
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ...interfaces import CriterionResult, Environment, Task, TraceStep, VerdictReport, Verifier
from .task import ChurnEnvironment


class ChurnVerifier(Verifier):
    def __init__(self, reference_date: date) -> None:
        self.reference_date = reference_date

    # ------------------------------------------------------------------
    # Ground-truth computation
    # ------------------------------------------------------------------

    def compute_ground_truth(self, db_path: Path) -> list[dict[str, Any]]:
        """
        Recompute the top-3 churn-risk customers from raw database data.

        Returns at most 3 dicts sorted by churn_score DESC then customer_id ASC.
        Each dict has keys: customer_id, prior_events, recent_events,
        failed_payments, churn_score.
        """
        ref = self.reference_date
        prior_start = (ref - timedelta(days=60)).isoformat()
        recent_start = (ref - timedelta(days=30)).isoformat()
        ref_str = ref.isoformat()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """
            WITH active_cust AS (
                SELECT customer_id FROM subscriptions WHERE status = 'active'
            ),
            prior AS (
                SELECT customer_id, COALESCE(SUM(event_count), 0) AS cnt
                FROM usage_events
                WHERE event_date >= :prior_start AND event_date < :recent_start
                GROUP BY customer_id
            ),
            recent AS (
                SELECT customer_id, COALESCE(SUM(event_count), 0) AS cnt
                FROM usage_events
                WHERE event_date >= :recent_start AND event_date < :ref_date
                GROUP BY customer_id
            ),
            fails AS (
                SELECT customer_id, COUNT(*) AS cnt
                FROM payments
                WHERE status = 'failed'
                  AND payment_date >= :prior_start
                  AND payment_date < :ref_date
                GROUP BY customer_id
            )
            SELECT
                ac.customer_id,
                COALESCE(p.cnt, 0)  AS prior_events,
                COALESCE(r.cnt, 0)  AS recent_events,
                COALESCE(f.cnt, 0)  AS failed_payments,
                MAX(0.0,
                    (COALESCE(p.cnt, 0) * 1.0 - COALESCE(r.cnt, 0) * 1.0)
                    / MAX(COALESCE(p.cnt, 0) * 1.0, 1.0)
                ) * (1.0 + COALESCE(f.cnt, 0)) AS churn_score
            FROM active_cust ac
            LEFT JOIN prior p ON ac.customer_id = p.customer_id
            LEFT JOIN recent r ON ac.customer_id = r.customer_id
            LEFT JOIN fails  f ON ac.customer_id = f.customer_id
            ORDER BY churn_score DESC, ac.customer_id ASC
            LIMIT 3
            """,
            {"prior_start": prior_start, "recent_start": recent_start, "ref_date": ref_str},
        ).fetchall()
        conn.close()

        return [
            {
                "customer_id": row[0],
                "prior_events": row[1],
                "recent_events": row[2],
                "failed_payments": row[3],
                "churn_score": row[4],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Criterion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _criterion_well_formed(answer_path: Path) -> tuple[CriterionResult, list[dict], bool]:
        """Parse the submission and return (criterion, parsed entries, ok flag)."""
        if not answer_path.exists():
            return (
                CriterionResult("well_formed_json", False, "answer.json not found"),
                [],
                False,
            )
        try:
            payload = json.loads(answer_path.read_text())
            top = payload.get("top_churn_customers", [])
            ok = (
                isinstance(top, list)
                and len(top) == 3
                and all(
                    isinstance(e, dict)
                    and isinstance(e.get("customer_id"), int)
                    and isinstance(e.get("justification"), str)
                    for e in top
                )
            )
            detail = "Valid structure" if ok else "Missing/wrong fields or not exactly 3 entries"
            return CriterionResult("well_formed_json", ok, detail), top if ok else [], ok
        except json.JSONDecodeError as exc:
            return CriterionResult("well_formed_json", False, f"JSON parse error: {exc}"), [], False

    @staticmethod
    def _criterion_ids(submitted: list[dict], expected_ids: list[int]) -> CriterionResult:
        sub_ids = [e["customer_id"] for e in submitted]
        ok = set(sub_ids) == set(expected_ids)
        return CriterionResult(
            "correct_ids",
            ok,
            f"expected set {expected_ids}, got {sub_ids}",
        )

    @staticmethod
    def _criterion_order(submitted: list[dict], expected_ids: list[int]) -> CriterionResult:
        sub_ids = [e["customer_id"] for e in submitted]
        ok = sub_ids == expected_ids
        return CriterionResult(
            "correct_order",
            ok,
            f"expected order {expected_ids}, got {sub_ids}",
        )

    @staticmethod
    def _criterion_process(trace: list[TraceStep]) -> CriterionResult:
        """Verify the agent queried all three relevant tables."""
        sql_queries: list[str] = []
        for step in trace:
            if step.kind == "tool_call":
                content = step.content or {}
                if content.get("tool") == "run_sql":
                    q = content.get("arguments", {}).get("query", "")
                    if q:
                        sql_queries.append(q.lower())

        all_sql = " ".join(sql_queries)
        hit_usage = "usage_events" in all_sql
        hit_payments = "payments" in all_sql
        hit_subs = "subscriptions" in all_sql
        ok = hit_usage and hit_payments and hit_subs
        return CriterionResult(
            "queried_tables",
            ok,
            (
                f"usage_events={hit_usage}, "
                f"payments={hit_payments}, "
                f"subscriptions={hit_subs}"
            ),
        )

    # ------------------------------------------------------------------
    # Public evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        task: Task,
        env: Environment,
        trace: list[TraceStep],
    ) -> VerdictReport:
        assert isinstance(env, ChurnEnvironment), "ChurnVerifier requires ChurnEnvironment"

        ground_truth = self.compute_ground_truth(env.db_path)
        expected_ids = [r["customer_id"] for r in ground_truth]

        answer_path = env.working_dir / "answer.json"
        crit_wf, entries, wf_ok = self._criterion_well_formed(answer_path)
        crit_ids = self._criterion_ids(entries, expected_ids) if wf_ok else CriterionResult("correct_ids", False, "submission malformed")
        crit_order = self._criterion_order(entries, expected_ids) if wf_ok else CriterionResult("correct_order", False, "submission malformed")
        crit_process = self._criterion_process(trace)

        criteria = [crit_wf, crit_ids, crit_order, crit_process]
        passed = all(c.passed for c in criteria)
        score = sum(1 for c in criteria if c.passed) / len(criteria)

        return VerdictReport(
            instance_id=task.instance_id,
            passed=passed,
            score=score,
            criteria=criteria,
        )
