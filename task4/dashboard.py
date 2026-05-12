"""Task 4C — Flask observability dashboard.

Single page at http://localhost:5000 with five panels, all SQL-backed against
the existing `traces` and `eval_runs` tables. Zero JS; auto-refresh every 10s.

Run:
    python -m task4.dashboard
    # or for a one-shot terminal report (no Flask):
    python -m task4.dashboard --print

The panels mirror Task 4C's exact requirements from the brief:
  1. Per-agent token usage + cost over time
  2. Tool call success/failure rates with error categorisation
  3. P50/P95 latency per agent + per tool
  4. Average turns-to-completion for intake
  5. Eval scores over time (regression detection)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Force UTF-8 stdout for terminal report (Urdu / non-ASCII in trace inputs).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import shared.config  # noqa: F401 — loads .env
from shared.db import get_conn


# ---------- SQL helpers ----------

def _q(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def panel_tokens() -> list[dict]:
    return _q(
        """SELECT actor,
                  date(ts) AS day,
                  COUNT(*) AS calls,
                  SUM(input_tokens) AS tok_in,
                  SUM(output_tokens) AS tok_out,
                  ROUND(SUM(cost_usd), 5) AS cost_usd
           FROM traces
           WHERE actor_kind IN ('agent','llm')
           GROUP BY actor, date(ts)
           ORDER BY day DESC, cost_usd DESC"""
    )


def panel_tool_outcomes() -> list[dict]:
    return _q(
        """SELECT actor AS tool,
                  status,
                  COALESCE(error_class, '-') AS error_class,
                  COUNT(*) AS n
           FROM traces
           WHERE actor_kind = 'tool'
           GROUP BY tool, status, error_class
           ORDER BY tool, status, n DESC"""
    )


def panel_latency() -> list[dict]:
    """SQLite has no PERCENTILE_CONT; emulate via ordered window + NTILE."""
    return _q(
        """WITH ranked AS (
             SELECT actor, actor_kind, latency_ms,
                    ROW_NUMBER() OVER (PARTITION BY actor ORDER BY latency_ms) AS rn,
                    COUNT(*)    OVER (PARTITION BY actor) AS cnt
             FROM traces
             WHERE latency_ms > 0
           )
           SELECT actor, actor_kind, cnt AS samples,
                  MAX(CASE WHEN rn = CAST(cnt * 0.50 AS INT) + 1 THEN latency_ms END) AS p50_ms,
                  MAX(CASE WHEN rn = CAST(cnt * 0.95 AS INT) + 1 THEN latency_ms END) AS p95_ms,
                  MAX(latency_ms) AS max_ms
           FROM ranked
           GROUP BY actor, actor_kind, cnt
           ORDER BY actor_kind, p95_ms DESC NULLS LAST"""
    )


def panel_intake_completion() -> dict:
    """For each customer who hit at least 1 turn, count their turns and
    whether all 6 lead fields ended up populated."""
    rows = _q(
        """SELECT et.customer_id,
                  et.session_id,
                  COUNT(*) AS turns,
                  (SELECT COUNT(*) FROM leads l WHERE l.customer_id = et.customer_id) AS fields_collected
           FROM episodic_turns et
           WHERE et.role = 'user'
           GROUP BY et.customer_id, et.session_id
           ORDER BY et.customer_id"""
    )
    completed = [r for r in rows if r["fields_collected"] >= 6]
    incomplete = [r for r in rows if r["fields_collected"] < 6]
    avg_completed = (sum(r["turns"] for r in completed) / len(completed)) if completed else None
    return {
        "rows": rows,
        "completed_sessions": len(completed),
        "incomplete_sessions": len(incomplete),
        "avg_turns_to_completion": round(avg_completed, 2) if avg_completed is not None else None,
    }


def panel_eval_scores() -> list[dict]:
    rows = _q(
        """SELECT suite,
                  scenario_id,
                  json_extract(details_json, '$.dimension') AS dimension,
                  date(ts) AS day,
                  passed,
                  ROUND(score, 2) AS score
           FROM eval_runs
           ORDER BY ts DESC"""
    )
    # Roll up: per (dimension, day), pass rate.
    by_dim: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["dimension"] or r["suite"], r["day"])
        agg = by_dim.setdefault(key, {"dimension": key[0], "day": key[1],
                                       "passed": 0, "total": 0})
        agg["total"] += 1
        agg["passed"] += r["passed"]
    aggregates = sorted(
        by_dim.values(),
        key=lambda d: (d["day"], d["dimension"]),
        reverse=True,
    )
    for a in aggregates:
        a["pass_rate"] = round(a["passed"] / max(a["total"], 1), 2)
    return aggregates


# ---------- HTML rendering ----------

PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Agentic AI — Observability</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .sub { color: #888; font-size: 0.85rem; margin-bottom: 24px; }
  h2 { font-size: 1.05rem; margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 6px; }
  .desc { color: #666; font-size: 0.85rem; margin-bottom: 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }
  th { background: #fafafa; font-weight: 600; }
  tr:hover { background: #fafafa; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; }
  .ok      { background: #e6f4ea; color: #1e8e3e; }
  .err     { background: #fce8e6; color: #c5221f; }
  .neutral { background: #f1f3f4; color: #5f6368; }
  .kpi { display: inline-block; margin-right: 24px; }
  .kpi .v { font-size: 1.4rem; font-weight: 600; }
  .kpi .l { color: #666; font-size: 0.8rem; }
</style></head><body>
<h1>Agentic Operations Intelligence — Observability</h1>
<div class="sub">Live SQL view of traces + eval runs. Auto-refresh every 10s.</div>

<h2>1. Token usage + cost per agent</h2>
<div class="desc">Cumulative input/output tokens and USD cost by actor and day (Groq pricing).</div>
__TOKENS_TABLE__

<h2>2. Tool outcomes</h2>
<div class="desc">Every tool call lands here, categorised by status and error_class.</div>
__TOOLS_TABLE__

<h2>3. Latency (P50 / P95 / max ms)</h2>
<div class="desc">Percentiles per agent and per tool. Tail latency is the one that bites.</div>
__LATENCY_TABLE__

<h2>4. Intake sessions — turns to completion</h2>
<div class="desc">For each intake session: total user turns and how many of the 6 lead fields ended up captured.</div>
<div>
  <span class="kpi"><div class="v">__COMPLETED__</div><div class="l">completed sessions (all 6 fields)</div></span>
  <span class="kpi"><div class="v">__INCOMPLETE__</div><div class="l">incomplete sessions</div></span>
  <span class="kpi"><div class="v">__AVG_TURNS__</div><div class="l">avg turns to completion</div></span>
</div>
__INTAKE_TABLE__

<h2>5. Eval pass-rate by dimension (over time)</h2>
<div class="desc">Eval scores per dimension, grouped by day. Watch for regressions on prompt changes.</div>
__EVAL_TABLE__

</body></html>
"""


def _render_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols: list of (key, label, css_class)."""
    if not rows:
        return '<div class="desc"><em>(no rows yet)</em></div>'
    header = "".join(f"<th>{label}</th>" for _, label, _ in cols)
    body_rows = []
    for r in rows:
        cells = []
        for key, _, css in cols:
            v = r.get(key)
            if v is None:
                v = "-"
            if css == "pill":
                cls = "ok" if v in ("ok", "success", "created") else ("err" if v in ("error", "fail") else "neutral")
                cells.append(f'<td><span class="pill {cls}">{v}</span></td>')
            else:
                cells.append(f'<td class="{css}">{v}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_page() -> str:
    tokens = panel_tokens()
    tools = panel_tool_outcomes()
    latency = panel_latency()
    intake = panel_intake_completion()
    evals = panel_eval_scores()

    avg_turns_display = (
        str(intake["avg_turns_to_completion"])
        if intake["avg_turns_to_completion"] is not None else "-"
    )

    replacements = {
        "__TOKENS_TABLE__": _render_table(
            tokens,
            [("day", "day", ""), ("actor", "actor", ""),
             ("calls", "calls", "num"), ("tok_in", "in tokens", "num"),
             ("tok_out", "out tokens", "num"), ("cost_usd", "cost USD", "num")],
        ),
        "__TOOLS_TABLE__": _render_table(
            tools,
            [("tool", "tool", ""), ("status", "status", "pill"),
             ("error_class", "error_class", ""), ("n", "n", "num")],
        ),
        "__LATENCY_TABLE__": _render_table(
            latency,
            [("actor", "actor", ""), ("actor_kind", "kind", ""),
             ("samples", "n", "num"), ("p50_ms", "P50 ms", "num"),
             ("p95_ms", "P95 ms", "num"), ("max_ms", "max ms", "num")],
        ),
        "__COMPLETED__": str(intake["completed_sessions"]),
        "__INCOMPLETE__": str(intake["incomplete_sessions"]),
        "__AVG_TURNS__": avg_turns_display,
        "__INTAKE_TABLE__": _render_table(
            intake["rows"],
            [("customer_id", "customer", ""), ("session_id", "session", ""),
             ("turns", "user turns", "num"), ("fields_collected", "fields/6", "num")],
        ),
        "__EVAL_TABLE__": _render_table(
            evals,
            [("day", "day", ""), ("dimension", "dimension", ""),
             ("passed", "passed", "num"), ("total", "total", "num"),
             ("pass_rate", "pass rate", "num")],
        ),
    }
    html = PAGE_TEMPLATE
    for marker, value in replacements.items():
        html = html.replace(marker, value)
    return html


# ---------- terminal mode (no Flask, useful for the Loom) ----------

def print_report() -> None:
    print("\n=== Token usage + cost ===")
    for r in panel_tokens():
        print(f"  {r['day']}  {r['actor']:<25}  calls={r['calls']:<4}  "
              f"tok={r['tok_in']}/{r['tok_out']:<7}  ${r['cost_usd']:.5f}")

    print("\n=== Tool outcomes ===")
    for r in panel_tool_outcomes():
        print(f"  {r['tool']:<30}  status={r['status']:<8}  err={r['error_class']:<25}  n={r['n']}")

    print("\n=== Latency P50/P95 ===")
    for r in panel_latency():
        print(f"  {r['actor']:<30}  ({r['actor_kind']})  n={r['samples']:<4}  "
              f"P50={r['p50_ms']}ms  P95={r['p95_ms']}ms  max={r['max_ms']}ms")

    intake = panel_intake_completion()
    print(f"\n=== Intake completion ===")
    print(f"  completed sessions: {intake['completed_sessions']}")
    print(f"  incomplete sessions: {intake['incomplete_sessions']}")
    print(f"  avg turns to completion: {intake['avg_turns_to_completion']}")

    print("\n=== Eval pass-rate by dimension ===")
    for r in panel_eval_scores():
        print(f"  {r['day']}  {r['dimension']:<14}  {r['passed']}/{r['total']}  ({r['pass_rate']*100:.0f}%)")


# ---------- entry point ----------

def main() -> int:
    parser = argparse.ArgumentParser(prog="task4.dashboard")
    parser.add_argument("--print", action="store_true",
                        help="render a terminal report instead of launching Flask")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("DASHBOARD_PORT", "5000")))
    args = parser.parse_args()

    if args.print:
        print_report()
        return 0

    from flask import Flask
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_page()

    @app.route("/api/tokens")
    def api_tokens():
        return json.dumps(panel_tokens()), 200, {"Content-Type": "application/json"}

    @app.route("/api/tools")
    def api_tools():
        return json.dumps(panel_tool_outcomes()), 200, {"Content-Type": "application/json"}

    @app.route("/api/latency")
    def api_latency():
        return json.dumps(panel_latency()), 200, {"Content-Type": "application/json"}

    @app.route("/api/intake")
    def api_intake():
        return json.dumps(panel_intake_completion()), 200, {"Content-Type": "application/json"}

    @app.route("/api/evals")
    def api_evals():
        return json.dumps(panel_eval_scores()), 200, {"Content-Type": "application/json"}

    print(f"Dashboard on http://localhost:{args.port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
