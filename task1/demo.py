"""End-to-end demo. Runs one orchestration and prints the full trace.

Usage:
    python -m task1.demo
    python -m task1.demo "What's blocking line 2?"
"""
from __future__ import annotations

import json
import sys

import shared.config  # noqa: F401 — loads .env
from shared.db import get_conn
from shared.ids import new_id

from .registry import build_system

DEFAULT_OBJECTIVE = (
    "Give me this morning's operations briefing — cover both stock levels for "
    "our key materials and any production bottlenecks today."
)


def _print_trace(trace_id: str) -> None:
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT ts, actor, actor_kind, status, error_class,
                      input_tokens, output_tokens, latency_ms, cost_usd
               FROM traces WHERE trace_id = ? ORDER BY ts""",
            (trace_id,),
        ).fetchall()
    finally:
        conn.close()

    print(f"\n--- Trace ({len(rows)} spans) ---")
    total_tok_in = total_tok_out = 0
    total_cost = 0.0
    for r in rows:
        total_tok_in += r["input_tokens"] or 0
        total_tok_out += r["output_tokens"] or 0
        total_cost += r["cost_usd"] or 0.0
        err = f" err={r['error_class']}" if r["error_class"] else ""
        print(
            f"  {r['ts']}  {r['actor']:<12} {r['actor_kind']:<6} "
            f"{r['status']:<10} tok={r['input_tokens']}/{r['output_tokens']:<5} "
            f"{r['latency_ms']}ms  ${r['cost_usd']:.5f}{err}"
        )
    print(f"--- Totals: tokens={total_tok_in}/{total_tok_out}  cost=${total_cost:.5f} ---\n")


def main() -> int:
    objective = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OBJECTIVE
    trace_id = new_id("trace")
    supervisor = build_system()

    print(f"\nObjective: {objective}\n(trace_id={trace_id})\n")
    result = supervisor.run(objective, trace_id=trace_id)

    print("=== Final answer ===")
    print(result["answer"])
    print("\n=== Plan ===")
    print(json.dumps(result["plan"], indent=2))
    print("\n=== Subtask results (truncated) ===")
    for r in result["results"]:
        print(f"  {r['agent']}.{r['intent']} -> {r['status']} (attempts={r['attempts']})")
    _print_trace(trace_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
