"""Run the 5 orchestration scenarios, assert, print a report, persist to eval_runs.

Usage:
    python -m task1.eval.runner

Exits with code 0 if all pass, 1 otherwise. Designed to be runnable headlessly.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import shared.config  # noqa: F401 — loads .env
from shared.db import get_conn
from shared.ids import new_id
from shared.llm import LLMClient
from shared.messages import AgentMessage  # noqa: F401  (used implicitly)

from ..agents.supervisor import SupervisorAgent
from .scenarios import SCENARIOS, Scenario


SUITE = "orchestration"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _agents_invoked(trace_id: str) -> dict[str, list[str]]:
    """Return {agent_name: [statuses, ...]} for all `agent`-kind spans in this trace."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT actor, status FROM traces WHERE trace_id = ? AND actor_kind = 'agent'",
            (trace_id,),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["actor"], []).append(r["status"])
    return out


def _evaluate(scenario: Scenario, trace_id: str, run_output: dict) -> list[CheckResult]:
    invoked = _agents_invoked(trace_id)
    # The supervisor itself shows up in agent spans; ignore it for the routing check.
    called = {a for a in invoked.keys() if a != "supervisor"}
    checks: list[CheckResult] = []

    # 1) expected_agents_called must be a subset of called
    missing = scenario.expected_agents_called - called
    checks.append(
        CheckResult(
            "expected_agents_called",
            not missing,
            f"called={sorted(called)} missing={sorted(missing)}",
        )
    )

    # 2) forbidden_agents_called must not appear
    forbidden_hit = scenario.forbidden_agents_called & called
    checks.append(
        CheckResult(
            "forbidden_agents_not_called",
            not forbidden_hit,
            f"forbidden_hit={sorted(forbidden_hit)}",
        )
    )

    # 3) subtask outcomes line up
    subtask_statuses = {
        r["agent"]: r["status"] for r in run_output["results"]
    }
    expected = scenario.expected_subtask_statuses
    mismatches = {
        a: (expected[a], subtask_statuses.get(a))
        for a in expected
        if subtask_statuses.get(a) != expected[a]
    }
    checks.append(
        CheckResult(
            "subtask_statuses_match",
            not mismatches,
            f"mismatches={mismatches}" if mismatches else "ok",
        )
    )

    # 4) must_produce_final_answer
    if scenario.must_produce_final_answer:
        ans = run_output.get("answer") or ""
        checks.append(
            CheckResult(
                "final_answer_non_empty",
                len(ans.strip()) >= 10,
                f"len={len(ans)}",
            )
        )

    return checks


def _persist(run_id: str, scenario: Scenario, checks: list[CheckResult]) -> None:
    conn = get_conn()
    try:
        passed = all(c.passed for c in checks)
        score = sum(1 for c in checks if c.passed) / max(len(checks), 1)
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs
               (run_id, scenario_id, suite, ts, passed, score, details_json, prompt_version)
               VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?)""",
            (
                run_id,
                scenario.id,
                SUITE,
                1 if passed else 0,
                score,
                json.dumps([c.__dict__ for c in checks]),
                "sup.v1",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_all() -> int:
    llm = LLMClient()
    run_id = new_id("run")
    print(f"\n=== Task 1 — orchestration eval | run_id={run_id} ===\n")
    total_pass = 0
    started = time.perf_counter()

    for scenario in SCENARIOS:
        trace_id = new_id("trace")
        print(f"[{scenario.id}] {scenario.objective}")
        registry = scenario.registry_factory(llm)
        supervisor = SupervisorAgent(llm=llm, registry=registry)
        try:
            output = supervisor.run(scenario.objective, trace_id=trace_id)
        except Exception as e:
            print(f"  CRASHED: {type(e).__name__}: {e}")
            _persist(run_id, scenario,
                     [CheckResult("no_crash", False, f"{type(e).__name__}: {e}")])
            continue

        checks = _evaluate(scenario, trace_id, output)
        _persist(run_id, scenario, checks)
        scenario_passed = all(c.passed for c in checks)
        total_pass += int(scenario_passed)
        marker = "PASS" if scenario_passed else "FAIL"
        print(f"  {marker}")
        for c in checks:
            print(f"    - {c.name}: {'ok' if c.passed else 'FAIL'} — {c.detail}")
        if scenario.notes:
            print(f"    note: {scenario.notes}")
        print()

    elapsed = time.perf_counter() - started
    print(f"=== {total_pass}/{len(SCENARIOS)} scenarios passed in {elapsed:.1f}s ===")
    return 0 if total_pass == len(SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(run_all())
