"""
Anthropic-API-driven agent.

Reads ANTHROPIC_API_KEY and ANTHROPIC_MODEL from the environment.
If no key is present, falls back to BaselineAgent with a clear log message.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..interfaces import Agent, Environment, Task, Tool, TraceStep

log = logging.getLogger(__name__)


class ModelAgent(Agent):
    """
    Drives the Anthropic Messages API using native tool use.

    The loop:
      1. Send system + user turn (task prompt) + tool schemas to the API.
      2. The model either replies with text (end_turn) or requests tool calls.
      3. For each tool call: execute the tool, record call+result in the trace,
         append a tool_result block to the conversation.
      4. Repeat until the model calls submit_answer, hits end_turn, or max_steps.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_anthropic_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert Tool objects to the Anthropic tool-definition schema."""
        result = []
        for t in tools:
            s = dict(t.schema)
            description = s.pop("description", t.name)
            result.append(
                {
                    "name": t.name,
                    "description": description,
                    "input_schema": s,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Agent.run
    # ------------------------------------------------------------------

    def run(
        self,
        task: Task,
        tools: list[Tool],
        env: Environment,
        max_steps: int = 20,
    ) -> list[TraceStep]:
        if not self.api_key:
            log.warning(
                "ANTHROPIC_API_KEY not set — falling back to BaselineAgent."
            )
            from .baseline import BaselineAgent

            return BaselineAgent().run(task, tools, env, max_steps)

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Install the 'anthropic' package to use ModelAgent: "
                "pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        tool_map = {t.name: t for t in tools}
        anthropic_tools = self._build_anthropic_tools(tools)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.prompt}
        ]
        trace: list[TraceStep] = []
        step = 0
        submitted = False

        while step < max_steps and not submitted:
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=anthropic_tools,
                messages=messages,
            )

            # Record any text/thinking in the trace
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    trace.append(
                        TraceStep(step=step, kind="thought", content={"text": block.text})
                    )

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                # Unexpected stop — exit gracefully
                log.warning("Unexpected stop_reason=%s", response.stop_reason)
                break

            # Execute each requested tool call
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                name: str = block.name
                args: dict[str, Any] = dict(block.input)

                trace.append(
                    TraceStep(
                        step=step,
                        kind="tool_call",
                        content={"tool": name, "arguments": args},
                    )
                )

                tool_obj = tool_map.get(name)
                if tool_obj is None:
                    result: Any = {"error": f"Unknown tool: {name!r}"}
                else:
                    result = tool_obj.execute(env, **args)

                trace.append(
                    TraceStep(
                        step=step,
                        kind="tool_result",
                        content={"tool": name, "result": result},
                    )
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    }
                )

                if name == "submit_answer":
                    submitted = True

            # Advance the conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            step += 1

        return trace
