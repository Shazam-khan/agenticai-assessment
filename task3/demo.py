"""End-to-end HITL pause/resume demo.

Two scenarios run back-to-back:

  Scenario A — sub-threshold PO: auto-executes.
  Scenario B — over-threshold PO: pauses, demo approves out-of-band, resumes.

Both go through Task 1's supervisor + the rewired Production Agent + the
`create_purchase_order` tool. Real Groq LLM (planner + synthesis).

Run:
    python -m task3.demo
"""
from __future__ import annotations

import json
import sys

import shared.config  # noqa: F401 — loads .env
from shared import hitl
from shared.db import get_conn
from shared.ids import new_id

from task1.registry import build_system


def _separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _show_pos() -> None:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT po_id, material, quantity, supplier_id, urgency, status, confirmation_id "
            "FROM purchase_orders ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("  (no purchase orders yet)")
        return
    for r in rows:
        cid = r["confirmation_id"][:12] + "..." if r["confirmation_id"] else "-"
        print(f"  PO {r['po_id'][:12]}...  {r['material']:<16} qty={r['quantity']:<6} "
              f"sup={r['supplier_id']:<8} urgency={r['urgency']:<8} cid={cid}")


def _show_pending() -> None:
    rows = hitl.list_pending()
    pending = [r for r in rows if r["status"] == "pending"]
    if not pending:
        print("  (no pending confirmations)")
        return
    for r in pending:
        print(f"  PENDING cid={r['confirmation_id'][:16]}...  tool={r['tool_name']}")
        print(f"          inputs={r['inputs_json']}")
        print(f"          reason={r['reason']}")


def scenario_a(supervisor) -> None:
    _separator("Scenario A — sub-threshold PO (auto-executes)")
    objective = "Place a standard-urgency order for 300 units of glycerin from SUP-002."
    trace_id = new_id("traceA")
    print(f"OBJECTIVE: {objective}\n(trace_id={trace_id})\n")
    result = supervisor.run(objective, trace_id=trace_id)
    print("Result status:", result["status"])
    print("Plan subtasks:", [s["intent"] for s in result["plan"]["subtasks"]])
    print("\nFinal answer:")
    print(result.get("answer", "(no answer)"))
    print("\nPurchase orders after Scenario A:")
    _show_pos()


def scenario_b(supervisor) -> None:
    _separator("Scenario B — over-threshold PO (HITL pause/resume)")
    objective = "Place a high-urgency PO for 1500 units of glycerin with SUP-002."
    trace_id = new_id("traceB")
    print(f"OBJECTIVE: {objective}\n(trace_id={trace_id})\n")

    # ---- pause ----
    result = supervisor.run(objective, trace_id=trace_id)
    print(f"Result status: {result['status']}")
    if result["status"] != "needs_human":
        print("Did NOT pause as expected. Bailing out of Scenario B.")
        print(json.dumps(result, indent=2, default=str)[:600])
        return
    cid = result["pending"][0]["confirmation_id"]
    print(f"Paused. confirmation_id={cid[:24]}...")
    print("Reason:", result["pending"][0]["reason"])
    print("\nPending confirmations on disk:")
    _show_pending()

    # ---- out-of-band approval ----
    print("\n>> Approving via shared.hitl.approve(...) "
          "(production: 'python -m task3.cli approve <id>')")
    ok = hitl.approve(cid, approver="demo_user")
    print(f"   approval returned: {ok}")

    # ---- resume ----
    print("\n>> Resuming...")
    result2 = supervisor.resume(trace_id)
    print(f"Resume status: {result2['status']}")
    if result2["status"] == "ok":
        print("\nFinal answer:")
        print(result2["answer"])
    else:
        print(json.dumps(result2, indent=2, default=str)[:600])

    print("\nPurchase orders after Scenario B:")
    _show_pos()
    print("\nConfirmation status after resume:")
    row = hitl.get_status(cid)
    print(f"  {cid[:16]}...  status={row['status']}  approved_by={row['approved_by']}")


def main() -> int:
    supervisor = build_system()
    scenario_a(supervisor)
    scenario_b(supervisor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
