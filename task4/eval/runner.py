"""Headless eval runner for Task 4B.

Runs the 15 scenarios in [task4/eval/scenarios.py](scenarios.py), assembles a
per-dimension score report, and persists each scenario's result to `eval_runs`
so the dashboard (Task 4C) can chart eval-scores-over-time.

Usage:
    python -m task4.eval.runner                 # all dimensions
    python -m task4.eval.runner --suite language
    python -m task4.eval.runner --judge-only    # only naturalness scenarios
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# On Windows cp1252 stdout chokes on Urdu glyphs in scenario text; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import shared.config  # noqa: F401 — loads .env
from shared.db import get_conn
from shared.ids import new_id
from shared.llm import LLMClient
from shared.messages import AgentMessage
from shared.tools.intake_tools import get_lead_fields, get_session_language

from task4.intake_v2 import IntakeAgentV2

from .scenarios import SCENARIOS, AssertResult, FinalState, Scenario

SUITE = "intake_v2"
PROMPT_VERSION = "intake.t4.v1"

# Default: production reasoning model. The eval is the ceiling-of-quality
# proof — we run it against the SAME model the agent uses in production so the
# scores reflect what users actually see. If the daily TPD cap on 70B is hit,
# set TASK4_EVAL_MODEL=llama-3.1-8b-instant as a fallback (lower scores expected
# on multi-field-extraction edge cases — see Task 4 README).
import os
EVAL_MODEL = os.getenv("TASK4_EVAL_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("TASK4_JUDGE_MODEL", EVAL_MODEL)

# ---------- LLM-as-judge ----------

JUDGE_SYSTEM = """You are an evaluator scoring the conversation naturalness of
an intake agent. The agent is supposed to ask for AT MOST ONE missing field
per turn, never dump the full list, never re-ask for a field the customer
already gave.

Reply with a single JSON object: {"passed": <bool>, "reason": "<one line>"}.
"""


def _judge_naturalness(llm: LLMClient, scenario_id: str, transcript: list[dict],
                       trace_id: str) -> AssertResult:
    """One LLM call per naturalness scenario; the model returns pass/fail."""
    from pydantic import BaseModel

    class _Judgment(BaseModel):
        passed: bool
        reason: str

    convo = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in transcript)
    user = (
        f"Scenario: {scenario_id}\n\n"
        f"Transcript:\n{convo}\n\n"
        "Did the agent ask for at most one missing field in each of its turns, "
        "without dumping the full list?"
    )
    try:
        judgment, _ = llm.complete_json(
            system=JUDGE_SYSTEM,
            user=user,
            schema=_Judgment,
            trace_id=trace_id,
            prompt_version="judge.v1",
            model=JUDGE_MODEL,
        )
    except Exception as e:
        return AssertResult(False, f"judge error: {e}")
    return AssertResult(judgment.passed, judgment.reason)


# ---------- per-scenario execution ----------

def _read_final_state(scenario: Scenario, customer_id: str, session_id: str,
                      transcript: list[dict], agent_outputs: list[dict],
                      trace_id: str) -> FinalState:
    leads_row = get_lead_fields(customer_id=customer_id, trace_id=trace_id).output
    lang_row = get_session_language(session_id=session_id, trace_id=trace_id).output
    return FinalState(
        customer_id=customer_id,
        session_id=session_id,
        transcript=transcript,
        agent_outputs=agent_outputs,
        leads=leads_row.get("collected", {}) if leads_row else {},
        session_language=lang_row.get("language") if lang_row else None,
    )


def _run_scenario(scenario: Scenario, llm: LLMClient, agent: IntakeAgentV2,
                  run_id: str) -> tuple[bool, list[AssertResult]]:
    customer_id = f"eval_{scenario.id}"
    session_id = f"eval_{scenario.id}_session"
    trace_id = new_id("eval")

    transcript: list[dict] = []
    agent_outputs: list[dict] = []

    for turn_text in scenario.turns:
        transcript.append({"role": "user", "content": turn_text})
        msg = AgentMessage(
            trace_id=trace_id, from_agent="user", to_agent="intake",
            intent="intake_turn",
            payload={"customer_id": customer_id, "session_id": session_id,
                     "channel": "eval", "content": turn_text},
        )
        reply = agent.handle(msg)
        if reply.status != "success":
            transcript.append({"role": "agent",
                                "content": f"[ERROR: {reply.error}]"})
            agent_outputs.append({"error": reply.error.model_dump() if reply.error else None})
            continue
        transcript.append({"role": "agent", "content": reply.payload["response"]})
        agent_outputs.append(reply.payload)

    state = _read_final_state(scenario, customer_id, session_id,
                                transcript, agent_outputs, trace_id)

    results: list[AssertResult] = []
    for spec in scenario.asserts:
        if spec.name == "naturalness_judge":
            results.append(_judge_naturalness(llm, scenario.id, transcript, trace_id))
        else:
            results.append(spec.check(state))

    passed = all(r.passed for r in results)
    return passed, results


# ---------- persistence + reporting ----------

def _persist(run_id: str, scenario: Scenario, passed: bool,
             results: list[AssertResult]) -> None:
    score = sum(1 for r in results if r.passed) / max(len(results), 1)
    details = {
        "dimension": scenario.dimension,
        "checks": [{"name": s.name, "passed": r.passed, "detail": r.detail}
                   for s, r in zip(scenario.asserts, results)],
    }
    conn = get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs
               (run_id, scenario_id, suite, ts, passed, score, details_json, prompt_version)
               VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?)""",
            (run_id, scenario.id, SUITE, 1 if passed else 0, score,
             json.dumps(details), PROMPT_VERSION),
        )
        conn.commit()
    finally:
        conn.close()


def run(suite_filter: str | None = None, judge_only: bool = False) -> int:
    llm = LLMClient(model=EVAL_MODEL)
    agent = IntakeAgentV2(llm=llm)
    run_id = new_id("run")
    started = time.perf_counter()

    print(f"\n=== Task 4B eval | run_id={run_id} | model={EVAL_MODEL} ===\n")

    by_dim: dict[str, list[tuple[str, bool, list[AssertResult]]]] = {}
    for scenario in SCENARIOS:
        if suite_filter and scenario.dimension != suite_filter:
            continue
        if judge_only and scenario.dimension != "naturalness":
            continue
        print(f"[{scenario.dimension}/{scenario.id}] "
              f"{scenario.turns[0][:60] + ('...' if len(scenario.turns[0]) > 60 else '')}")
        passed, results = _run_scenario(scenario, llm, agent, run_id)
        _persist(run_id, scenario, passed, results)
        marker = "PASS" if passed else "FAIL"
        print(f"  {marker}")
        for spec, r in zip(scenario.asserts, results):
            print(f"    - {spec.name}: {'ok' if r.passed else 'FAIL'} -- {r.detail}")
        by_dim.setdefault(scenario.dimension, []).append((scenario.id, passed, results))
        print()

    elapsed = time.perf_counter() - started
    print("=== Summary ===")
    overall_pass = 0
    overall_total = 0
    for dim, rows in by_dim.items():
        p = sum(1 for _, ok, _ in rows if ok)
        t = len(rows)
        overall_pass += p
        overall_total += t
        print(f"  {dim:<14} {p}/{t}")
    print(f"  {'TOTAL':<14} {overall_pass}/{overall_total}   ({elapsed:.1f}s)")
    return 0 if overall_pass == overall_total else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="task4.eval.runner")
    p.add_argument("--suite", choices=["language", "extraction", "hallucination",
                                          "adversarial", "naturalness"])
    p.add_argument("--judge-only", action="store_true")
    args = p.parse_args()
    return run(suite_filter=args.suite, judge_only=args.judge_only)


if __name__ == "__main__":
    sys.exit(main())
