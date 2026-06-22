"""Tool implementations for the churn task environment."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .interfaces import Environment, Tool

if TYPE_CHECKING:
    from .tasks.churn.task import ChurnEnvironment


# ---------------------------------------------------------------------------
# SQL safety guards
# ---------------------------------------------------------------------------

_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|pragma|replace|upsert|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _validate_sql(query: str) -> str | None:
    """Return an error message if *query* is not a safe SELECT, else None."""
    if not _SELECT_RE.match(query):
        return "Only SELECT statements are permitted."
    if _FORBIDDEN_RE.search(query):
        return "Query contains a forbidden keyword."
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class ListTablesTool(Tool):
    """Return the names of all user tables in the task database."""

    name = "list_tables"
    schema = {
        "type": "object",
        "description": "Return the names of all tables available in the database.",
        "properties": {},
        "required": [],
    }

    def execute(self, env: "ChurnEnvironment", **kwargs: Any) -> dict:
        try:
            conn = sqlite3.connect(str(env.db_path))
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            conn.close()
            return {"tables": [r[0] for r in rows]}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


class DescribeTableTool(Tool):
    """Return column definitions for a named table."""

    name = "describe_table"
    schema = {
        "type": "object",
        "description": "Return column definitions (name, type, nullability, primary-key flag) for a table.",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to describe.",
            }
        },
        "required": ["table_name"],
    }

    def execute(self, env: "ChurnEnvironment", table_name: str = "", **kwargs: Any) -> dict:
        try:
            conn = sqlite3.connect(str(env.db_path))
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            conn.close()
            if not rows:
                return {"error": f"Table '{table_name}' not found."}
            columns = [
                {
                    "cid": r[0],
                    "name": r[1],
                    "type": r[2],
                    "notnull": bool(r[3]),
                    "pk": bool(r[5]),
                }
                for r in rows
            ]
            return {"table": table_name, "columns": columns}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


class RunSqlTool(Tool):
    """Execute a read-only SELECT query and return results."""

    name = "run_sql"
    schema = {
        "type": "object",
        "description": (
            "Execute a SELECT query against the task database and return rows. "
            "Only SELECT statements are allowed."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": "A valid SQL SELECT statement.",
            }
        },
        "required": ["query"],
    }

    def execute(self, env: "ChurnEnvironment", query: str = "", **kwargs: Any) -> dict:
        err = _validate_sql(query)
        if err:
            return {"error": err}
        try:
            conn = sqlite3.connect(str(env.db_path))
            conn.execute("PRAGMA query_only = ON")
            cur = conn.execute(query)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [list(row) for row in cur.fetchall()]
            conn.close()
            return {"columns": columns, "rows": rows, "row_count": len(rows)}
        except sqlite3.Error as exc:
            return {"error": f"SQL error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


class ReadFileTool(Tool):
    """Read a text file from the agent's sandbox working directory."""

    name = "read_file"
    schema = {
        "type": "object",
        "description": "Read a file from the agent's working directory.",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename (relative, within the working directory) to read.",
            }
        },
        "required": ["filename"],
    }

    def execute(self, env: Environment, filename: str = "", **kwargs: Any) -> dict:
        try:
            sandbox = env.working_dir.resolve()
            target = (env.working_dir / filename).resolve()
            # Strict sandbox check — no path traversal
            if not str(target).startswith(str(sandbox) + "/") and target != sandbox:
                return {"error": "Access denied: path is outside the working directory."}
            if not target.exists():
                return {"error": f"File '{filename}' not found."}
            if not target.is_file():
                return {"error": f"'{filename}' is not a file."}
            return {"content": target.read_text()}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


class SubmitAnswerTool(Tool):
    """Write the final answer to answer.json in the working directory."""

    name = "submit_answer"
    schema = {
        "type": "object",
        "description": (
            "Submit the final answer. Writes answer.json to the working directory. "
            "Call this exactly once when you are confident in your ranking."
        ),
        "properties": {
            "answer": {
                "type": "object",
                "description": (
                    "Must contain 'top_churn_customers': a list of exactly 3 objects "
                    "each with 'customer_id' (int) and 'justification' (str), "
                    "ordered from highest to lowest churn risk."
                ),
                "properties": {
                    "top_churn_customers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "customer_id": {"type": "integer"},
                                "justification": {"type": "string"},
                            },
                            "required": ["customer_id", "justification"],
                        },
                        "minItems": 3,
                        "maxItems": 3,
                    }
                },
                "required": ["top_churn_customers"],
            }
        },
        "required": ["answer"],
    }

    def execute(self, env: Environment, answer: dict | None = None, **kwargs: Any) -> dict:
        if answer is None:
            return {"error": "Missing 'answer' argument."}
        try:
            answer_path = env.working_dir / "answer.json"
            answer_path.write_text(json.dumps(answer, indent=2))
            return {"status": "submitted", "path": str(answer_path)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


def default_tools() -> list[Tool]:
    """Return the standard tool set for the churn task."""
    return [
        ListTablesTool(),
        DescribeTableTool(),
        RunSqlTool(),
        ReadFileTool(),
        SubmitAnswerTool(),
    ]
