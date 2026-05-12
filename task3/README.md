# Task 3 — Tool design, reliability & self-correction

Four tools, structured-error validation, idempotency, a circuit breaker, and the most architecturally consequential piece in the assessment: a **persist-and-exit HITL mechanism** the agent loop yields to and resumes from.

## The HITL pause/resume answer

The brief calls out:
> *"the human-in-the-loop requirement is not just a UI feature — it is a fundamental architectural question: how does an autonomous agent pause, request confirmation, and resume?"*

Our answer:

```
        Supervisor.run("Order 1500 units of glycerin...")
             │
             ├─► plan ──► [{place_order, material=glycerin, quantity=1500, ...}]
             │
             ▼
        Production Agent._handle_place_order
             │
             ▼
        create_purchase_order(quantity=1500, confirmed=False)
             │
             │  quantity > THRESHOLD and not confirmed
             ▼
        hitl.create_pending(...)
             │
             ▼
   ToolResult(status="needs_confirmation", confirmation_id=C)
             │
             ▼
   AgentMessage(status="needs_human", payload={confirmation_id: C, ...})
             │
             ▼
   Supervisor.run() sees needs_human, persists paused_runs(trace_id, objective),
   returns {status: "needs_human", pending: [{confirmation_id: C, reason: ...}]}.
   The agent loop EXITS — no `input()`, no blocked thread.

   ──────── out-of-band ────────
   $ python -m task3.cli approve <C>   # OR: shared.hitl.approve(C, ...)
   ──────────────────────────────

   Supervisor.resume(trace_id):
        │
        ├─► loads objective from paused_runs
        ├─► run() again — same trace_id
        │    - planner emits the same plan
        │    - existing PO id (idempotent)? short-circuits
        │    - create_purchase_order(quantity=1500):
        │        * sees confirmation C is `status='approved'`
        │        * inserts purchase_orders row
        │        * marks C `status='executed'`
        │        * returns ToolResult.ok({po_id, status: "created"})
        ▼
   Final answer synthesised, paused_runs.status='resumed'.
```

The key properties:
- **No in-memory state survives across the pause** — paused_runs has only `(trace_id, objective)`. The replay-from-scratch + idempotency-keys combo handles everything else.
- **The confirmation_id is derived from the inputs**, so re-submitting before approval lands on the same pending row (no duplicate pending), and re-submitting *after* approval executes (no double-execution either — `po_id` is also derived from inputs+day).
- **Approval is out-of-band** — a CLI command, an API endpoint, a Slack bot, anything that flips the row to 'approved'.

Demonstrated end-to-end in [task3/demo.py](demo.py). Scenario B output:

```
Result status: needs_human
Paused. confirmation_id=53e969f3302a92a0...
Reason: Quantity 1500.0 exceeds threshold 500 for material 'glycerin'. Human approval required.
>> Approving via shared.hitl.approve(...)
   approval returned: True
>> Resuming...
Resume status: ok
Final answer: The high-urgency PO for 1500 units of glycerin with SUP-002 has been successfully created. ...
Confirmation status after resume: status=executed approved_by=demo_user
```

## Idempotency strategy per tool

| Tool | Key | Behaviour on repeat |
|---|---|---|
| `get_stock_levels` | n/a (pure read) | Same inputs → same outputs. No side effect. |
| `create_purchase_order` | `sha256(material + quantity + supplier_id + YYYY-MM-DD_utc)` as `po_id` | Same order today → `status='existing'`, original PO returned. Same order tomorrow → new PO (a legitimate repeat). |
| `get_production_schedule` | n/a (pure read) | Same inputs → same outputs. |
| `flag_bottleneck` | `sha256(order_id + reason.lower().strip() + YYYY-MM-DD_utc)` as `alert_id` | Same flag today → `status='existing'`. Same flag tomorrow OR different reason → new alert. |

Pending confirmations have their own key: `sha256(tool_name + canonical_json(inputs))`. Same inputs → same `confirmation_id`, no duplicate pending row.

## Circuit-breaker state machine

```
                ┌──────────┐
                │  closed  │
                └────┬─────┘
                     │ N consecutive failures
                     ▼
                ┌──────────┐
                │   open   │── deny all calls ── return ToolResult.err("circuit_open")
                └────┬─────┘
                     │ cooldown elapsed
                     ▼
                ┌──────────┐
                │ half_open │── allow ONE probe (other callers denied)
                └────┬─────┘
                ┌────┴────┐
        success │         │ failure
                ▼         ▼
            closed     open  (cooldown doubled, capped at 5 min)
```

Per-tool state in the `circuit_state` table. Threshold + initial cooldown configurable via env (`CIRCUIT_BREAKER_FAILURES`, `CIRCUIT_BREAKER_COOLDOWN_SECONDS`). The exponential cooldown means a chronically-broken tool stops getting hammered.

Tests in [tests/test_task3_circuit.py](../tests/test_task3_circuit.py) cover all four transitions.

## Adversarial inputs — what the test suite proves

[tests/test_task3_tools.py](../tests/test_task3_tools.py): ≥3 cases per tool. **Every malformed input returns a structured `ToolResult.error`, never an exception** (brief hard requirement #1).

| Tool | Flavour | Example | Result |
|---|---|---|---|
| `get_stock_levels` | malformed | `materials=None` | error `schema_violation` |
| `get_stock_levels` | malformed | `materials=[1,2,3]` | error `schema_violation` |
| `get_stock_levels` | boundary  | `materials=[]` | error `schema_violation` (min_length=1) |
| `get_stock_levels` | plausible | `materials=["glycerine"]` (typo) | ok, surfaces in `unknown` list |
| `create_purchase_order` | malformed | `quantity="ten"` | error `schema_violation` |
| `create_purchase_order` | boundary | `quantity=0` | error `schema_violation` (gt=0) |
| `create_purchase_order` | boundary | `quantity=500` (at threshold) | ok, auto-creates |
| `create_purchase_order` | boundary | `quantity=501` (just above) | needs_confirmation |
| `create_purchase_order` | plausible | `supplier_id="SUP-9999"` | error `unknown_supplier` |
| `create_purchase_order` | idempotency | same call twice | first creates, second returns `existing` |
| `get_production_schedule` | malformed | `date_range=None` | error `schema_violation` |
| `get_production_schedule` | malformed | non-ISO strings | error `schema_violation` |
| `get_production_schedule` | boundary | start > end | error `schema_violation` |
| `flag_bottleneck` | malformed | `severity="catastrophic"` | error `invalid_severity` |
| `flag_bottleneck` | malformed | `order_id=None` | error `schema_violation` |
| `flag_bottleneck` | boundary | `reason=""` | error `schema_violation` (min_length=1) |
| `flag_bottleneck` | plausible | `order_id="PO-0000"` (nonexistent) | error `unknown_order` |
| `flag_bottleneck` | idempotency | same flag same day | first creates, second returns `existing` |

## Cross-task integration

Production Agent ([task1/agents/production.py](../task1/agents/production.py)) now uses the tools in its own flow:
- `_handle_check_schedule` reads via `tools.get_production_schedule` (traced like any other tool span); after the LLM identifies bottlenecks, it calls `tools.flag_bottleneck` for each one (idempotent, so re-runs are safe).
- `_handle_place_order` is the new branch — calls `tools.create_purchase_order`, propagates `needs_confirmation` up as `AgentMessage.status="needs_human"`.

Supervisor ([task1/agents/supervisor.py](../task1/agents/supervisor.py)) gained:
- `place_order` routing rule in `PLANNER_SYSTEM` with two concrete examples.
- A new `needs_human` outcome in `_dispatch_with_retry` (no retries — propagate cleanly).
- `run()` short-circuits to a `{status: "needs_human", pending: [...]}` return when any subtask is paused.
- `resume(trace_id)` replays the objective with idempotency-driven dedupe.

Task 1's 5/5 eval still passes — the planner's `place_order` rule didn't regress routing (verified). The fix to the planner prompt that this surfaced (date_range emitted as natural-language placeholders) is documented in the supervisor diff.

## Run it

```powershell
# Unit tests for Task 3 (30 tests, ~10s, no API key)
.venv\Scripts\python.exe -m pytest tests/test_task3_tools.py tests/test_task3_hitl.py tests/test_task3_circuit.py tests/test_task3_supervisor_pause.py -v

# Re-confirm Tasks 1 and 2 didn't regress
.venv\Scripts\python.exe -m pytest tests/ -v
.venv\Scripts\python.exe -m task1.eval.runner

# Real-LLM HITL demo (Scenarios A + B end-to-end)
.venv\Scripts\python.exe -m task3.demo

# CLI surface
.venv\Scripts\python.exe -m task3.cli list
.venv\Scripts\python.exe -m task3.cli approve <confirmation_id>
.venv\Scripts\python.exe -m task3.cli resume <trace_id>

# Inspect side-effects
.venv\Scripts\python.exe -c "from shared.db import get_conn; c=get_conn(); print('POs:'); [print(dict(r)) for r in c.execute('SELECT po_id, material, quantity, urgency, status FROM purchase_orders')]; print('\\nALERTS:'); [print(dict(r)) for r in c.execute('SELECT alert_id, order_id, reason, severity FROM alerts')]; print('\\nCIRCUITS:'); [print(dict(r)) for r in c.execute('SELECT * FROM circuit_state')]"
Get-Content data/notifications.log
```

## Tradeoffs

| Choice | Win | Lose / upgrade path |
|---|---|---|
| **Resume = full plan replay, not surgical resume** | Simplest correct mechanism; survives process restarts (no in-memory state); idempotency was a hard req anyway, so we get this for free. | Re-runs every LLM call in the plan, paying tokens again. Upgrade: cache plan + completed-subtask outputs keyed by `(trace_id, span_path)`, skip already-traced work. |
| **confirmation_id derived from inputs, not random** | Idempotent resubmission; the same call always finds the same pending row; dedupe is free. | Can't issue "two orders of the same shape" without varying *some* input (we use ordering-day for the common case). Upgrade: optional `idempotency_token` arg for callers that need finer control. |
| **Circuit-breaker scope = per tool-name** | Simple state machine, predictable behaviour. | One supplier going down kills all PO writes. Upgrade: per `(tool, key_arg)` circuits with fallback suppliers. |
| **Notification side-effect = log-file append** | Zero dependency on Slack/PagerDuty; legibly real; swap point is a one-line function (`production_tools._notify`). | Nothing actually wakes a human during the assessment. Upgrade: replace `_notify` with a webhook + paging routing. |
| **Cooldown doubles on each open (capped at 5min)** | Backs off on chronic failures; matches common patterns. | Can mask "tool recovered" cases where the circuit is still long-cooled. Upgrade: `task3.cli reset-circuit <tool>` admin command. |
| **Per-day idempotency window for POs** | Same order today → one PO; same order tomorrow → legitimate new PO. | The window is a calendar day in UTC, so a midnight roll-over could create a new PO 1 second after the previous one. Upgrade: explicit `ordering_window_hours` arg or caller-provided idempotency_token. |
