"""ChurnTask and ChurnEnvironment."""
from __future__ import annotations

import shutil
import tempfile
from datetime import date
from pathlib import Path

from ...interfaces import Environment, Task
from .build_db import build_db

_PROMPT_TEMPLATE = """\
You are a data analyst with access to a customer database.

Your task: identify the TOP 3 CUSTOMERS MOST AT RISK OF CHURN.

=== Churn-Risk Definition ===
Only customers with an ACTIVE subscription (subscriptions.status = 'active') qualify.

For each qualifying customer compute:

  prior_events  = SUM(event_count) in usage_events
                  WHERE event_date >= '{prior_start}' AND event_date < '{recent_start}'

  recent_events = SUM(event_count) in usage_events
                  WHERE event_date >= '{recent_start}' AND event_date < '{ref_date}'

  failed_payments = COUNT(*) in payments
                    WHERE status = 'failed'
                      AND payment_date >= '{prior_start}'
                      AND payment_date < '{ref_date}'

  usage_decline_ratio = MAX(0, (prior_events - recent_events) / MAX(prior_events, 1))

  churn_score = usage_decline_ratio * (1 + failed_payments)

Rank by churn_score DESC; break ties by customer_id ASC.

Reference date (treat as "today"): {ref_date}

=== Your output ===
Call the `submit_answer` tool with:
{{
  "top_churn_customers": [
    {{"customer_id": <int>, "justification": "<one-line reason>"}},
    {{"customer_id": <int>, "justification": "<one-line reason>"}},
    {{"customer_id": <int>, "justification": "<one-line reason>"}}
  ]
}}
The list must be ordered from HIGHEST to LOWEST churn risk.

Use the available tools to explore the database schema and compute the answer.
Do NOT guess — the correct answer depends entirely on the data in the database.
"""


class ChurnEnvironment(Environment):
    """Isolated environment containing the task SQLite database and sandbox dir."""

    def __init__(self, working_dir: Path, db_path: Path, reference_date: date) -> None:
        self._working_dir = working_dir
        self.db_path = db_path
        self.reference_date = reference_date
        self._owned_tmp: Path | None = None  # set by factory if we created the tmpdir

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    def teardown(self) -> None:
        """Remove the temp working directory and its contents."""
        root = self._owned_tmp or self._working_dir
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


class ChurnTask(Task):
    """
    SQL data-analysis task: identify the top-3 churn-risk customers.

    Each *seed* produces a deterministically different database, so running
    multiple seeds gives independent task instances.
    """

    def __init__(self, seed: int = 42, reference_date: date | None = None) -> None:
        self.seed = seed
        self._reference_date = reference_date or date(2024, 3, 1)

    @property
    def instance_id(self) -> str:
        return f"churn-seed{self.seed}"

    @property
    def reference_date(self) -> date:
        return self._reference_date

    @property
    def prompt(self) -> str:
        from datetime import timedelta

        ref = self._reference_date
        prior_start = ref - timedelta(days=60)
        recent_start = ref - timedelta(days=30)
        return _PROMPT_TEMPLATE.format(
            ref_date=ref.isoformat(),
            prior_start=prior_start.isoformat(),
            recent_start=recent_start.isoformat(),
        )

    def build_env(self) -> ChurnEnvironment:
        """Create a temp directory, build the DB, and return the environment."""
        tmp = Path(tempfile.mkdtemp(prefix=f"agentic_eval_{self.instance_id}_"))
        db_path = tmp / "churn.db"
        ref_date = build_db(db_path, seed=self.seed, reference_date=self._reference_date)

        env = ChurnEnvironment(
            working_dir=tmp,
            db_path=db_path,
            reference_date=ref_date,
        )
        env._owned_tmp = tmp
        return env
