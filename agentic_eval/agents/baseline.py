"""
Baseline scripted agent.

Performs a deterministic query sequence that correctly solves the churn task
without any external API calls.  Used as:
  - the offline / no-key fallback
  - a sanity-check that a legitimate agent run passes all verifier criteria
"""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from ..interfaces import Agent, Environment, Task, Tool, TraceStep

if TYPE_CHECKING:
    from ..tasks.churn.task import ChurnEnvironment


class BaselineAgent(Agent):
    """
    Deterministic scripted agent that correctly solves the churn task.

    Calls tools in a logical order that mirrors what a competent analyst
    would do, ensuring the process criterion is satisfied alongside the
    outcome criteria.
    """

    def run(
        self,
        task: Task,
        tools: list[Tool],
        env: Environment,
        max_steps: int = 20,
    ) -> list[TraceStep]:
        # Obtain the reference date directly from the environment (the baseline
        # has access to env, unlike a real model that only sees the prompt).
        churn_env: "ChurnEnvironment" = env  # type: ignore[assignment]
        ref = churn_env.reference_date
        prior_start = (ref - timedelta(days=60)).isoformat()
        recent_start = (ref - timedelta(days=30)).isoformat()
        ref_str = ref.isoformat()

        tool_map = {t.name: t for t in tools}
        trace: list[TraceStep] = []
        step = 0

        def call(name: str, **kwargs: object) -> object:
            nonlocal step
            trace.append(
                TraceStep(
                    step=step,
                    kind="tool_call",
                    content={"tool": name, "arguments": kwargs},
                )
            )
            result = tool_map[name].execute(env, **kwargs)
            trace.append(
                TraceStep(
                    step=step,
                    kind="tool_result",
                    content={"tool": name, "result": result},
                )
            )
            step += 1
            return result

        # 1. Discover schema
        call("list_tables")
        for tbl in ("customers", "subscriptions", "usage_events", "payments"):
            call("describe_table", table_name=tbl)

        # 2. List active customers (satisfies subscriptions query criterion)
        call(
            "run_sql",
            query=(
                "SELECT c.customer_id, c.name "
                "FROM customers c "
                "JOIN subscriptions s ON c.customer_id = s.customer_id "
                "WHERE s.status = 'active' "
                "ORDER BY c.customer_id"
            ),
        )

        # 3. Spot-check usage trend (satisfies usage_events criterion)
        call(
            "run_sql",
            query=(
                f"SELECT customer_id, SUM(event_count) AS cnt "
                f"FROM usage_events "
                f"WHERE event_date >= '{recent_start}' AND event_date < '{ref_str}' "
                f"GROUP BY customer_id ORDER BY cnt DESC LIMIT 5"
            ),
        )

        # 4. Spot-check failed payments (satisfies payments criterion)
        call(
            "run_sql",
            query=(
                f"SELECT customer_id, COUNT(*) AS fails "
                f"FROM payments "
                f"WHERE status = 'failed' "
                f"  AND payment_date >= '{prior_start}' "
                f"  AND payment_date < '{ref_str}' "
                f"GROUP BY customer_id ORDER BY fails DESC"
            ),
        )

        # 5. Compute full churn scores and pick top 3
        churn_result = call(
            "run_sql",
            query=f"""
SELECT
    ac.customer_id,
    COALESCE(p.cnt, 0)  AS prior_events,
    COALESCE(r.cnt, 0)  AS recent_events,
    COALESCE(f.cnt, 0)  AS failed_payments,
    MAX(0.0,
        (COALESCE(p.cnt, 0) * 1.0 - COALESCE(r.cnt, 0) * 1.0)
        / MAX(COALESCE(p.cnt, 0) * 1.0, 1.0)
    ) * (1.0 + COALESCE(f.cnt, 0)) AS churn_score
FROM (
    SELECT customer_id FROM subscriptions WHERE status = 'active'
) ac
LEFT JOIN (
    SELECT customer_id, COALESCE(SUM(event_count), 0) AS cnt
    FROM usage_events
    WHERE event_date >= '{prior_start}' AND event_date < '{recent_start}'
    GROUP BY customer_id
) p ON ac.customer_id = p.customer_id
LEFT JOIN (
    SELECT customer_id, COALESCE(SUM(event_count), 0) AS cnt
    FROM usage_events
    WHERE event_date >= '{recent_start}' AND event_date < '{ref_str}'
    GROUP BY customer_id
) r ON ac.customer_id = r.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS cnt
    FROM payments
    WHERE status = 'failed'
      AND payment_date >= '{prior_start}'
      AND payment_date < '{ref_str}'
    GROUP BY customer_id
) f ON ac.customer_id = f.customer_id
ORDER BY churn_score DESC, ac.customer_id ASC
LIMIT 3
""",
        )

        # 6. Build and submit the final answer
        top_3 = []
        result_dict = churn_result  # type: ignore[assignment]
        if isinstance(result_dict, dict) and "rows" in result_dict:
            for row in result_dict["rows"][:3]:
                cid, prior, recent, failed, score = row
                top_3.append(
                    {
                        "customer_id": int(cid),
                        "justification": (
                            f"Usage dropped from {prior} to {recent} events "
                            f"({int((1-(recent/max(prior,1)))*100)}% decline); "
                            f"{failed} failed payment(s) in window"
                        ),
                    }
                )

        call("submit_answer", answer={"top_churn_customers": top_3})
        return trace
