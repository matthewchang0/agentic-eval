"""Core abstractions for the agentic evaluation framework."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CriterionResult:
    """Pass/fail result for a single grading criterion."""

    name: str
    passed: bool
    detail: str


@dataclass
class VerdictReport:
    """Full evaluation verdict for one task instance."""

    instance_id: str
    passed: bool
    score: float  # [0, 1]
    criteria: list[CriterionResult] = field(default_factory=list)


@dataclass
class TraceStep:
    """One recorded step in an agent's execution trace."""

    step: int
    kind: str  # "thought" | "tool_call" | "tool_result"
    content: Any  # must be JSON-serialisable


class Tool(ABC):
    """A callable tool exposed to the agent."""

    name: str   # class-level constant, override in subclass
    schema: dict  # JSON Schema (type: object) for this tool's input arguments

    @abstractmethod
    def execute(self, env: "Environment", **kwargs: Any) -> Any:
        """
        Execute the tool inside *env*.

        Must never raise — return structured error dicts instead so the agent
        loop can relay the error without crashing.
        """


class Environment(ABC):
    """Sandboxed execution context for one task instance."""

    @property
    @abstractmethod
    def working_dir(self) -> Path:
        """The agent's writable sandbox directory."""

    @abstractmethod
    def teardown(self) -> None:
        """Release all resources (temp files, DB connections, etc.)."""


class Task(ABC):
    """A reproducible, deterministically-generatable evaluation task."""

    @property
    @abstractmethod
    def instance_id(self) -> str:
        """Unique identifier for this instance (e.g. 'churn-seed42')."""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Natural-language instructions given to the agent."""

    @abstractmethod
    def build_env(self) -> Environment:
        """Materialise a fresh, isolated environment for this task."""


class Agent(ABC):
    """An agent that attempts to complete a task via tool calls."""

    @abstractmethod
    def run(
        self,
        task: Task,
        tools: list[Tool],
        env: Environment,
        max_steps: int,
    ) -> list[TraceStep]:
        """
        Drive the agent until it submits an answer or hits *max_steps*.

        The implementation is responsible for both directing the agent and
        executing tool calls — it must record every tool_call and tool_result
        as TraceSteps.
        """


class Verifier(ABC):
    """Deterministic grader for a completed agent run."""

    @abstractmethod
    def evaluate(
        self,
        task: Task,
        env: Environment,
        trace: list[TraceStep],
    ) -> VerdictReport:
        """
        Grade the run.

        Ground truth must be recomputed from raw data — never read from a
        file the agent could have touched.
        """
