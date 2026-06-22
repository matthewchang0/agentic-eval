"""
CLI runner: python -m agentic_eval.run

Flags
-----
--agent   {baseline,model}   which agent to use (default: baseline)
--n       INT                number of task instances / seeds (default: 5)
--out     PATH               report output path (default: report.json)
--seed    INT                starting seed (default: 0); instances use seeds [seed, seed+n)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .agents.baseline import BaselineAgent
from .agents.model import ModelAgent
from .interfaces import VerdictReport
from .tasks.churn.task import ChurnTask
from .tasks.churn.verifier import ChurnVerifier
from .tools import default_tools
from .trace import save_trace

_MAX_STEPS = 25


def run_instance(
    agent: BaselineAgent | ModelAgent,
    task: ChurnTask,
    tools: list,
    max_steps: int = _MAX_STEPS,
) -> tuple[list, VerdictReport]:
    env = task.build_env()
    try:
        trace = agent.run(task, tools, env, max_steps)
        verifier = ChurnVerifier(task.reference_date)
        verdict = verifier.evaluate(task, env, trace)
        return trace, verdict
    finally:
        env.teardown()


def _defect_taxonomy(verdicts: list[VerdictReport]) -> dict[str, int]:
    taxonomy: dict[str, int] = {}
    for v in verdicts:
        if v.passed:
            continue
        for c in v.criteria:
            if not c.passed:
                taxonomy[c.name] = taxonomy.get(c.name, 0) + 1
    return taxonomy


def _print_table(verdicts: list[VerdictReport], criterion_rates: dict[str, float]) -> None:
    pass_rate = sum(1 for v in verdicts if v.passed) / len(verdicts)
    print()
    print("=" * 54)
    print(f"  Instances:  {len(verdicts)}")
    print(f"  Pass rate:  {pass_rate:.1%}")
    print()
    print("  Per-criterion pass rates:")
    for name, rate in criterion_rates.items():
        bar = "#" * int(rate * 20)
        print(f"    {name:<28} {rate:5.1%}  [{bar:<20}]")
    print()
    taxonomy = _defect_taxonomy(verdicts)
    if taxonomy:
        print("  Failure taxonomy:")
        for name, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
            print(f"    {name:<28} {count} failure(s)")
    print("=" * 54)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the agentic_eval churn task harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
            # Run 5 baseline instances and write report.json
            python -m agentic_eval.run --agent baseline --n 5

            # Run 3 instances with the Anthropic model agent
            ANTHROPIC_API_KEY=sk-... python -m agentic_eval.run --agent model --n 3
            """
        ),
    )
    parser.add_argument("--agent", choices=["baseline", "model"], default="baseline")
    parser.add_argument("--n", type=int, default=5, metavar="N", help="number of task instances")
    parser.add_argument("--seed", type=int, default=0, help="starting seed")
    parser.add_argument("--out", type=Path, default=Path("report.json"), help="report output path")
    parser.add_argument("--traces-dir", type=Path, default=None, help="directory to save per-instance traces")

    args = parser.parse_args(argv)

    agent: BaselineAgent | ModelAgent
    if args.agent == "model":
        agent = ModelAgent()
    else:
        agent = BaselineAgent()

    tools = default_tools()
    verdicts: list[VerdictReport] = []

    if args.traces_dir:
        args.traces_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        seed = args.seed + i
        task = ChurnTask(seed=seed)
        print(f"[{i+1}/{args.n}] {task.instance_id} ...", end=" ", flush=True)
        trace, verdict = run_instance(agent, task, tools)
        verdicts.append(verdict)

        if args.traces_dir:
            save_trace(trace, args.traces_dir / f"{task.instance_id}_trace.json")

        status = "PASS" if verdict.passed else f"FAIL (score={verdict.score:.2f})"
        print(status)

    # Build report
    all_criterion_names: list[str] = (
        [c.name for c in verdicts[0].criteria] if verdicts else []
    )
    criterion_rates: dict[str, float] = {}
    for name in all_criterion_names:
        passes = sum(
            1
            for v in verdicts
            for c in v.criteria
            if c.name == name and c.passed
        )
        criterion_rates[name] = passes / len(verdicts)

    mean_steps = 0.0  # traces not preserved here; placeholder

    report = {
        "agent": args.agent,
        "n_instances": len(verdicts),
        "pass_rate": sum(1 for v in verdicts if v.passed) / len(verdicts),
        "per_criterion_pass_rates": criterion_rates,
        "defect_taxonomy": _defect_taxonomy(verdicts),
        "instances": [
            {
                "instance_id": v.instance_id,
                "passed": v.passed,
                "score": v.score,
                "criteria": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in v.criteria
                ],
            }
            for v in verdicts
        ],
    }

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {args.out}")
    _print_table(verdicts, criterion_rates)

    return 0 if all(v.passed for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
