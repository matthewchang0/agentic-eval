"""Helpers for serialising and deserialising execution traces."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from .interfaces import TraceStep


def _default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def save_trace(trace: list[TraceStep], path: Path) -> None:
    """Serialise *trace* as JSON and write to *path*."""
    payload = {
        "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
        "steps": [
            {"step": s.step, "kind": s.kind, "content": s.content}
            for s in trace
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=_default))


def load_trace(path: Path) -> list[TraceStep]:
    """Read a trace from a JSON file written by :func:`save_trace`."""
    data = json.loads(path.read_text())
    return [
        TraceStep(step=s["step"], kind=s["kind"], content=s["content"])
        for s in data["steps"]
    ]
