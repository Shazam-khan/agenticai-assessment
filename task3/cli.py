"""HITL approval CLI — the out-of-band side of the pause/resume mechanism.

Usage:
    python -m task3.cli list
    python -m task3.cli approve <confirmation_id>
    python -m task3.cli reject  <confirmation_id>
    python -m task3.cli resume  <trace_id>

This is the documented interface for a human to unblock a paused run.
Production would wrap the same `shared.hitl` calls behind a web UI or Slack bot.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import shared.config  # noqa: F401 — loads .env
from shared import hitl


def _age(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return "?"
    delta = datetime.now(timezone.utc) - dt
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h"


def cmd_list(_args: argparse.Namespace) -> int:
    rows = hitl.list_pending()
    if not rows:
        print("(no pending confirmations)")
        return 0
    print(f"{'STATUS':<10} {'TOOL':<25} {'AGE':<6} {'CONFIRMATION_ID':<20} INPUTS")
    print("-" * 100)
    for r in rows:
        cid_short = r["confirmation_id"][:16]
        inputs = r["inputs_json"]
        if len(inputs) > 60:
            inputs = inputs[:57] + "..."
        print(f"{r['status']:<10} {r['tool_name']:<25} {_age(r['created_at']):<6} "
              f"{cid_short:<20} {inputs}")
    return 0


def _resolve(cid: str) -> str | None:
    """Allow short prefix matching for convenience."""
    rows = hitl.list_pending()
    matches = [r["confirmation_id"] for r in rows
               if r["confirmation_id"].startswith(cid)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"ambiguous prefix '{cid}', matches {len(matches)} rows", file=sys.stderr)
        return None
    # Fall through: maybe full id and not in 'pending' anymore.
    row = hitl.get_status(cid)
    return cid if row else None


def cmd_approve(args: argparse.Namespace) -> int:
    cid = _resolve(args.confirmation_id)
    if not cid:
        print(f"no confirmation matching '{args.confirmation_id}'", file=sys.stderr)
        return 1
    ok = hitl.approve(cid, approver=args.approver)
    if not ok:
        row = hitl.get_status(cid)
        print(f"could not approve {cid[:16]}; current status="
              f"{row['status'] if row else 'missing'}", file=sys.stderr)
        return 1
    print(f"approved {cid}")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    cid = _resolve(args.confirmation_id)
    if not cid:
        print(f"no confirmation matching '{args.confirmation_id}'", file=sys.stderr)
        return 1
    ok = hitl.reject(cid, approver=args.approver)
    if not ok:
        print(f"could not reject {cid[:16]}", file=sys.stderr)
        return 1
    print(f"rejected {cid}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    # Lazy import to avoid loading the LLM client just for `list`.
    from task1.registry import build_system
    supervisor = build_system()
    result = supervisor.resume(args.trace_id)
    print(json.dumps({
        "status": result.get("status"),
        "answer": result.get("answer", "")[:500],
        "pending": result.get("pending"),
    }, indent=2))
    return 0 if result.get("status") in ("ok", "success") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="task3.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show pending confirmations")

    pa = sub.add_parser("approve", help="approve a pending confirmation")
    pa.add_argument("confirmation_id")
    pa.add_argument("--approver", default="cli_user")

    pr = sub.add_parser("reject", help="reject a pending confirmation")
    pr.add_argument("confirmation_id")
    pr.add_argument("--approver", default="cli_user")

    prs = sub.add_parser("resume", help="resume a paused supervisor run")
    prs.add_argument("trace_id")

    args = p.parse_args(argv)
    return {
        "list": cmd_list,
        "approve": cmd_approve,
        "reject": cmd_reject,
        "resume": cmd_resume,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
