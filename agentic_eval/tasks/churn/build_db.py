"""
Deterministic synthetic database builder for the churn task.

Schema
------
customers      — customer_id, name, email, signup_date
subscriptions  — subscription_id, customer_id, status, start_date, plan
usage_events   — event_id, customer_id, event_date, event_count
payments       — payment_id, customer_id, payment_date, amount, status

Churn-risk tiers (assigned by seed-shuffling customer IDs):
  HIGH_RISK   (ids 0–3 of shuffled list): active sub, large usage decline, 2–4 failed payments
  MEDIUM_RISK (ids 4–7):                  active sub, moderate decline, 1 failed payment
  LOW_RISK    (ids 8–13):                 active sub, flat/growing usage, no failed payments
  INACTIVE    (ids 14–19):                cancelled/paused sub, variable usage

Tiers guarantee the top-3 by churn score always come from HIGH_RISK customers,
but WHICH three and in what ORDER depend entirely on the seed — hardcoded answers
are wrong for all but a single seed.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    email        TEXT    NOT NULL,
    signup_date  TEXT    NOT NULL
);

CREATE TABLE subscriptions (
    subscription_id  INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    status           TEXT    NOT NULL CHECK(status IN ('active','cancelled','paused')),
    start_date       TEXT    NOT NULL,
    plan             TEXT    NOT NULL CHECK(plan IN ('basic','pro','enterprise'))
);

CREATE TABLE usage_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    event_date   TEXT    NOT NULL,
    event_count  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE payments (
    payment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    payment_date  TEXT    NOT NULL,
    amount        REAL    NOT NULL,
    status        TEXT    NOT NULL CHECK(status IN ('success','failed'))
);
"""

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina",
]
_LAST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Davis", "Miller",
    "Wilson", "Moore", "Taylor", "Anderson",
]
_PLANS = ["basic", "pro", "enterprise"]
_AMOUNTS = [9.99, 29.99, 99.99]

N_CUSTOMERS = 20


def build_db(path: Path, seed: int = 42, reference_date: date | None = None) -> date:
    """
    Build a SQLite database at *path* deterministically from *seed*.

    Returns the *reference_date* used for churn calculations (the "today"
    of the task).  Date ranges in the churn formula are relative to this date:
      - prior period:           [ref − 60, ref − 30)
      - recent period:          [ref − 30, ref)
      - failed payments window: [ref − 60, ref)
    """
    if reference_date is None:
        reference_date = date(2024, 3, 1)

    rng = random.Random(seed)

    prior_start = reference_date - timedelta(days=60)
    recent_start = reference_date - timedelta(days=30)

    # Shuffle IDs so risk-tier assignment varies with seed
    ids = list(range(1, N_CUSTOMERS + 1))
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    high_risk_ids = set(shuffled[:4])     # 4 high-risk — top 3 will always come from here
    medium_risk_ids = set(shuffled[4:8])
    low_risk_ids = set(shuffled[8:14])
    # shuffled[14:] → inactive (cancelled/paused subscription)

    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)

    # ---- customers ----
    seen_names: set[str] = set()
    for cid in range(1, N_CUSTOMERS + 1):
        while True:
            name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
            if name not in seen_names:
                seen_names.add(name)
                break
        email = f"user{cid}@example.com"
        signup_date = reference_date - timedelta(days=rng.randint(90, 400))
        conn.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            (cid, name, email, signup_date.isoformat()),
        )

    # ---- subscriptions ----
    for cid in range(1, N_CUSTOMERS + 1):
        if cid in high_risk_ids or cid in medium_risk_ids or cid in low_risk_ids:
            status = "active"
        else:
            status = rng.choice(["cancelled", "paused"])
        start_date = reference_date - timedelta(days=rng.randint(45, 365))
        plan = rng.choice(_PLANS)
        conn.execute(
            "INSERT INTO subscriptions (customer_id, status, start_date, plan) VALUES (?, ?, ?, ?)",
            (cid, status, start_date.isoformat(), plan),
        )

    # ---- usage_events & payments ----
    for cid in range(1, N_CUSTOMERS + 1):
        if cid in high_risk_ids:
            prior_base = rng.randint(30, 50)
            decline_pct = rng.uniform(0.70, 0.90)
            n_failed = rng.randint(2, 4)
        elif cid in medium_risk_ids:
            prior_base = rng.randint(15, 30)
            decline_pct = rng.uniform(0.40, 0.60)
            n_failed = 1
        elif cid in low_risk_ids:
            prior_base = rng.randint(10, 25)
            decline_pct = rng.uniform(-0.10, 0.25)  # flat / slight growth
            n_failed = 0
        else:  # inactive — may have some history
            prior_base = rng.randint(5, 20)
            decline_pct = rng.uniform(0.0, 0.60)
            n_failed = rng.randint(0, 2)

        recent_count = max(0, int(prior_base * (1.0 - decline_pct)))

        # Prior-period usage events
        for _ in range(prior_base):
            offset = rng.randint(0, 29)
            event_date = prior_start + timedelta(days=offset)
            conn.execute(
                "INSERT INTO usage_events (customer_id, event_date, event_count) VALUES (?,?,1)",
                (cid, event_date.isoformat()),
            )

        # Recent-period usage events
        for _ in range(recent_count):
            offset = rng.randint(0, 29)
            event_date = recent_start + timedelta(days=offset)
            conn.execute(
                "INSERT INTO usage_events (customer_id, event_date, event_count) VALUES (?,?,1)",
                (cid, event_date.isoformat()),
            )

        # Successful payments (2–5 per customer)
        for _ in range(rng.randint(2, 5)):
            offset = rng.randint(0, 59)
            pdate = prior_start + timedelta(days=offset)
            conn.execute(
                "INSERT INTO payments (customer_id, payment_date, amount, status) VALUES (?,?,?,'success')",
                (cid, pdate.isoformat(), rng.choice(_AMOUNTS)),
            )

        # Failed payments
        for _ in range(n_failed):
            offset = rng.randint(0, 59)
            pdate = prior_start + timedelta(days=offset)
            conn.execute(
                "INSERT INTO payments (customer_id, payment_date, amount, status) VALUES (?,?,?,'failed')",
                (cid, pdate.isoformat(), rng.choice(_AMOUNTS)),
            )

    conn.commit()
    conn.close()
    return reference_date
