# Architecture — Shared Primitives

> One system, four tasks. This document defines the contracts that all four tasks
> share. Each task refines its own slice; none of them get to reshape these
> primitives without updating this file.

## Stack

- **LLM:** Groq (Llama 3.3 70B Versatile for reasoning, Llama 3.1 8B Instant for cheap classification / judge calls). Justification: lowest latency on the market, low enough cost to run a thorough eval harness repeatedly, and we control reliability ourselves via a validation/retry layer rather than relying on the provider's tool-use polish.
- **Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2`. No extra API key, runs in-process, 384-dim vectors. Adequate for top-3 fact retrieval at this scale.
- **Datastore:** single SQLite file (`./data/agentic.db`) for all relational state (traces, memory, HITL queue, circuit-breaker state, eval results) + `chromadb` persistent client (`./data/chroma/`) for the vector side.
- **Language:** Python 3.11+, `pydantic` v2 for schemas, `pytest` for evals & unit tests.

## Repo layout

```
/shared/         # the primitives in this doc — schemas, db, trace logger, HITL queue
/task1/          # multi-agent orchestration
/task2/          # memory system
/task3/          # tool library + reliability
/task4/          # prompts, eval harness, observability dashboard
/data/           # sqlite + chroma (gitignored)
/tests/          # cross-task tests (per-task tests live under each /taskN/)
ARCHITECTURE.md  # this file
README.md        # root, with diagram + tradeoffs
```

---

## 1. Agent message envelope

Every inter-agent message is a `pydantic` model — never a raw string. This is the contract Task 1 builds on and Task 3 (HITL) extends with a `needs_confirmation` status.

```python
class AgentMessage(BaseModel):
    trace_id: str           # ties all spans of one objective together
    span_id: str            # this hop
    parent_span_id: str | None
    from_agent: str         # "supervisor" | "inventory" | "production" | "report" | "user"
    to_agent: str
    intent: Literal["check_stock", "check_schedule", "draft_report",
                    "flag_bottleneck", "synthesise", "clarify", "escalate"]
    payload: dict           # validated per-intent in /shared/intents.py
    status: Literal["pending", "success", "error", "needs_human", "timeout"]
    attempts: int = 0
    metadata: MessageMetadata  # tokens, latency_ms, model, prompt_version, cost_usd
    created_at: datetime
```

`MessageMetadata` is what makes Task 4C's dashboard possible — every agent hop carries its own cost/latency/token data, so observability is a side-effect of the envelope, not a separate concern.

## 2. Trace / log table

Single append-only table written by every agent and every tool call. Powers Task 1 req #3 (per-action logs) **and** Task 4C (token usage, latency percentiles, eval-over-time).

```sql
CREATE TABLE traces (
  span_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  parent_span_id TEXT,
  ts TEXT NOT NULL,                 -- ISO8601
  actor TEXT NOT NULL,              -- agent or tool name
  actor_kind TEXT NOT NULL,         -- 'agent' | 'tool' | 'llm' | 'eval'
  model TEXT,
  prompt_version TEXT,
  input_json TEXT,
  output_json TEXT,
  status TEXT NOT NULL,             -- success | error | needs_human | timeout
  error_class TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  latency_ms INTEGER,
  cost_usd REAL
);
CREATE INDEX idx_traces_trace ON traces(trace_id);
CREATE INDEX idx_traces_actor_ts ON traces(actor, ts);
```

P50/P95 latency, token totals, tool success rates, and cost are all SQL queries against this one table. No second observability store.

## 3. Memory tables (Task 2)

Three layers with **explicit promotion logic**:

```
raw turn      ─►  extract facts  ─►  semantic fact
(episodic)        (LLM call,         (structured row)
                   prompt_version    + embedding
                   tracked)
```

```sql
CREATE TABLE episodic_turns (
  turn_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  customer_id TEXT NOT NULL,
  channel TEXT,                     -- 'web' | 'whatsapp' | 'email'
  role TEXT NOT NULL,               -- 'user' | 'agent'
  content TEXT NOT NULL,
  ts TEXT NOT NULL
);

CREATE TABLE semantic_facts (
  fact_id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  category TEXT NOT NULL,           -- controlled enum: company_info | contact |
                                    -- product_need | quantity | timeline | budget |
                                    -- constraint | preference | other
  entity TEXT,                      -- subject the fact attaches to, e.g. 'Lumen Botanicals'
  value TEXT NOT NULL,              -- e.g. '10000 units in 30ml glass bottles'
  confidence REAL,
  source_turn_id TEXT,
  created_at TEXT NOT NULL,
  last_accessed_at TEXT NOT NULL,
  access_count INTEGER DEFAULT 0,
  dedupe_key TEXT NOT NULL UNIQUE   -- sha256(customer + category + entity + value_norm)
);
-- vector side: chromadb collection 'semantic_facts', id = fact_id

-- Why category/entity/value instead of subject/predicate/object: see Task 2 README.
-- An LLM-generated predicate vocab is brittle (wants_quantity_of vs needs_quantity
-- vs requested_volume). A small controlled `category` enum + free-text `entity`
-- and `value` keeps retrieval reliable and dedupe deterministic.

CREATE TABLE working_memory (
  session_id TEXT PRIMARY KEY,
  summary TEXT,
  summary_token_count INTEGER,
  last_updated_at TEXT
);
```

**Decay strategy:** retrieval score = `cosine_sim * decay(now - last_accessed)` where `decay(d) = 1.0` for `d ≤ 90d` and `0.5 ^ ((d - 90) / 90)` after. Facts are never deleted; old ones just rank below relevant fresh ones. `last_accessed_at` updates on every retrieval (this is the "promotion" signal — frequently-recalled facts stay sharp).

**Token budget enforcement:** before building a prompt, sum `working_memory.summary + top-3 facts + last N raw turns`. If over `TOKEN_BUDGET` (default 4000), summarise the oldest unsummarised turns into `working_memory`, drop the raw turns, retry.

## 4. HITL pause / resume primitive (Task 3)

This is the hard part of Task 3 and the brief flags it explicitly: "how does an autonomous agent pause, request confirmation, and resume?"

**Mechanism:** tools return a `ToolResult` discriminated union. When a tool needs confirmation, it does **not** execute the side effect — it persists a row and returns `needs_confirmation`.

```python
class ToolResult(BaseModel):
    status: Literal["ok", "error", "needs_confirmation"]
    output: dict | None
    error: ToolError | None
    confirmation: PendingConfirmation | None

class PendingConfirmation(BaseModel):
    confirmation_id: str    # = sha256(tool_name + canonical_json(inputs))
    tool_name: str
    inputs: dict
    reason: str             # "quantity 1200 exceeds threshold 500"
    expires_at: datetime
```

```sql
CREATE TABLE pending_confirmations (
  confirmation_id TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  inputs_json TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  approved_at TEXT,
  approved_by TEXT,
  status TEXT NOT NULL              -- pending | approved | rejected | expired | executed
);
```

**Implementation** (Task 3 final shape):

1. Tool returns `ToolResult.needs_confirmation(PendingConfirmation(...))`. The pending row is persisted via `shared.hitl.create_pending`.
2. The calling agent (e.g. `task1.agents.production.ProductionAgent._handle_place_order`) sees the `needs_confirmation` ToolResult and replies with `AgentMessage(status="needs_human", payload={confirmation_id, reason})`.
3. The supervisor's `_dispatch_with_retry` treats `needs_human` as a non-retryable propagation case (no retry, no escalate).
4. `supervisor.run()`, after executing the plan, detects any `needs_human` subtask, writes `paused_runs(trace_id, objective)`, and returns `{status: "needs_human", pending: [...]}` — **the agent loop terminates**. No `input()`. No thread blocked.
5. Approval is out-of-band: `python -m task3.cli approve <confirmation_id>` (or `shared.hitl.approve(cid, approver)` from any process). Flips the row to `approved`.
6. `supervisor.resume(trace_id)` reads the objective from `paused_runs` and calls `run()` with the same `trace_id`. The plan replays; the previously-paused tool now sees an `approved` row for the matching `confirmation_id` and executes; idempotency keys keep everything else as a no-op.

**Idempotency** falls out for free: `confirmation_id` is derived from inputs (`sha256(tool_name + canonical_json(inputs))`), so re-submitting the same `create_purchase_order` returns the same pending row. For executed POs, the `po_id` is also derived from inputs + day (`sha256(material + quantity + supplier_id + utc_day)`); the `UNIQUE` constraint on `purchase_orders.po_id` makes a duplicate execute a no-op that returns the original PO. Same for `flag_bottleneck` alerts.

## 5. Circuit breaker (Task 3)

Per-tool state machine in SQLite:

```sql
CREATE TABLE circuit_state (
  tool_name TEXT PRIMARY KEY,
  state TEXT NOT NULL,              -- 'closed' | 'open' | 'half_open'
  consecutive_failures INTEGER DEFAULT 0,
  opened_at TEXT,
  cooldown_seconds INTEGER DEFAULT 60
);
```

`closed` → 3 consecutive failures → `open` (tool calls short-circuit with structured error reported to Supervisor). After `cooldown_seconds`, the next call moves to `half_open`; success closes the circuit, failure re-opens with doubled cooldown.

## 6. Eval runner (Tasks 1 & 4B)

One shared runner, two consumers. Task 1 runs 5 orchestration scenarios; Task 4B runs the 5-dimension intake suite (≥3 cases each).

```python
class Scenario(BaseModel):
    id: str
    suite: str                  # 'orchestration' | 'language' | 'extraction' | ...
    input: dict
    expectations: list[Expectation]   # each is a callable check(trace, output) -> Pass | Fail(reason)
```

The runner:
1. Resets a clean DB scope (per-scenario `trace_id`).
2. Invokes the system under test.
3. Reads the resulting trace rows and the final output.
4. Evaluates expectations.
5. Writes results to `eval_runs` so Task 4C can chart eval-scores-over-time.

```sql
CREATE TABLE eval_runs (
  run_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  suite TEXT NOT NULL,
  ts TEXT NOT NULL,
  passed INTEGER NOT NULL,
  score REAL,
  details_json TEXT,
  prompt_version TEXT,
  PRIMARY KEY (run_id, scenario_id)
);
```

## 7. Observability surface (Task 4C)

No new store — just SQL views over the existing tables, served by a single Flask page ([task4/dashboard.py](task4/dashboard.py)). The same module also has a `--print` mode that renders a terminal report (useful for the Loom).

- **Token usage + cost:** `SELECT actor, date(ts), COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM traces WHERE actor_kind IN ('agent','llm') GROUP BY 1,2`
- **Tool success/failure:** `SELECT actor AS tool, status, COALESCE(error_class,'-'), COUNT(*) FROM traces WHERE actor_kind='tool' GROUP BY 1,2,3`
- **P50/P95 latency per actor:** ordered window over `latency_ms`, percentile via row-number trick (SQLite has no `PERCENTILE_CONT`).
- **Intake turns-to-completion:** `episodic_turns` (role='user') joined with `leads` count — sessions with all 6 fields collected are "complete"; average turns over the completed set.
- **Eval pass-rate by dimension:** `eval_runs` grouped by `json_extract(details_json,'$.dimension')` + day.

Routes:
- `/` — the dashboard (auto-refreshes every 10s via `<meta http-equiv="refresh">`)
- `/api/{tokens,tools,latency,intake,evals}` — JSON for programmatic clients

---

## Cross-task contract summary

| Concern              | Defined by | Consumed by |
|----------------------|------------|-------------|
| `AgentMessage`       | shared     | Task 1, Task 3 (extends with HITL status), Task 4 (logged) |
| `traces` table       | shared     | Task 1 logs, Task 3 logs, Task 4C dashboard |
| Memory tables        | Task 2     | Task 1 intake agent, Task 4B hallucination tests |
| HITL primitive       | Task 3     | Task 1 supervisor loop (must respect `paused` status) |
| Circuit breaker      | Task 3     | Task 1 supervisor (handles `tool_unavailable` errors) |
| Eval runner          | shared     | Task 1 (5 scenarios), Task 4B (5 dimensions × 3+ cases) |
| `eval_runs` table    | shared     | Task 4C chart |

## Named tradeoffs (will grow as we build)

1. **Groq over Claude/OpenAI.** Win: latency + cost for repeated eval runs. Lose: weaker native tool-calling; we compensate with our own validation/retry layer (which Task 3 requires anyway, so the "loss" is partly absorbed).
2. **SQLite + Chroma over Postgres+pgvector.** Win: zero setup, single-file demo. Lose: not horizontally scalable. Upgrade path: swap the `/shared/db.py` driver — schema is portable.
3. **Local sentence-transformers embeddings.** Win: no extra vendor, free, deterministic. Lose: weaker than `text-embedding-3-large` on nuanced semantic similarity. Upgrade path: `/shared/embeddings.py` is one function.
4. **Single trace table for agents + tools + LLM calls.** Win: one query language for all observability. Lose: row volume grows fast. Upgrade path: partition by date, or move to a columnar store.
5. **HITL via persistence + out-of-band approval CLI** (not blocking input). Win: works in any agent loop, async, multi-channel. Lose: requires the approver to know the CLI / endpoint exists — needs a notification layer for real use.
6. **Decay function is monotonic (not learned).** Win: simple, debuggable, no eval-leakage risk. Lose: may downweight a still-relevant fact a customer hasn't mentioned in months. Upgrade path: learn decay per `predicate` type.
7. **Prompt-engineered routing in the Supervisor planner.** Win: cheap, debuggable, no second model. Lose: fragile to model upgrades — the first real-LLM eval run caught the planner over-routing on a single-topic question (see task1/README.md). Upgrade path: a routing-classifier eval suite that runs on every prompt change and on model bumps; alert on regressions before they ship.
8. **Exact-text dedupe on facts** (Task 2) — `sha256(customer | category | entity | value.lower().strip())`. Win: deterministic, no LLM cost on every write, auditable. Lose: "10k units" and "10,000 units" become two rows. Upgrade: a normaliser pass in the extractor (canonicalise numbers + units before hashing), or a periodic semantic-dedupe job.
9. **Token count is `len(text) // 4`** (Task 2), not a real tokenizer. Win: zero dependency on Groq/OpenAI tokenizers; fast; deterministic. Lose: ~10% inaccuracy means we summarise slightly earlier or later than the true budget. Upgrade: switch `shared/memory/working.count_tokens` to `tiktoken` cl100k_base (close enough for Llama) — one line.
10. **Live fact extraction per user turn** (Task 2) — one fast-LLM call before the reasoning call. Win: cross-session demo works without an explicit session-end signal; the demo proves it. Lose: doubles the LLM call count per intake turn (fast model, ~$0.0001 each, but still). Upgrade: extraction can run async; the reasoning prompt only needs facts that are *already* indexed, not the ones from the current turn.
11. **Resume = full plan replay, not surgical resume** (Task 3). Win: simplest correct mechanism; no in-memory state survives process restarts; idempotency keys carry all the dedup load. Lose: replay re-runs every LLM call in the plan (planner + each specialist + synthesis), paying tokens again. Upgrade: cache per-span outputs keyed by `(trace_id, span_path)` and short-circuit on resume.
12. **HITL confirmation_id is derived from inputs, not random** (Task 3). Win: idempotent resubmission; same call always finds the same pending row; one approval cleans up any number of duplicate submissions. Lose: a caller can't issue "two intentional orders of the same shape" without varying *some* input — we use ordering-day to handle the common case. Upgrade: optional `idempotency_token` arg for callers that need finer-than-daily resolution.
13. **Circuit breaker scoped per tool-name** (Task 3), not per `(tool, args)`. Win: a small predictable state table; trivial to reason about. Lose: one bad supplier kills all PO writes. Upgrade: per-`(tool, key_arg)` circuits, or per-supplier fallback routing.
14. **Anti-batching is enforced in the agent loop, not just the prompt** (Task 4A). The agent persists only fields in `learned_keys ∩ called_keys` of the current turn — a misbehaving model can't write phantom fields or batch learned-but-uncommitted ones. Win: contract holds even if the model drifts; phantom/unrecorded counts are surfaced for the eval to assert against. Lose: rejects legitimate single-turn typos in the model's own JSON. Upgrade: a permissive mode that warns instead of dropping.
15. **Eval defaults to Llama 3.3 70B Versatile** (Task 4B) — the same model the production agent uses, so eval scores reflect what users see, not a development-floor proxy. 8B Instant is available as a fallback (`TASK4_EVAL_MODEL=llama-3.1-8b-instant`) for iteration when the daily TPD cap is hit. Win: ceiling-of-quality measurement. Lose: each full eval run consumes a significant chunk of the daily 70B token budget; can't re-run on every prompt tweak without quota planning. Upgrade: paid Groq tier removes the budget concern entirely; alternatively, cache scenario outputs keyed by `(prompt_version, scenario_id)` so unchanged scenarios skip re-execution.
16. **LLM-as-judge only for the naturalness rubric** (Task 4B). All other dimensions use deterministic Python checks (regex, set membership, table lookups). Win: cheap and reproducible everywhere except the one inherently qualitative dimension. Lose: judge variance can flake on the boundary. Upgrade: 3-sample judge with majority vote.
17. **Heuristic language detection** — Urdu Unicode block presence vs. Latin (Task 4A/B). Win: zero dependency, deterministic, fast. Lose: false-negative on Roman Urdu (Urdu typed in Latin script — treated as English). Upgrade: fastText langid or a fast LLM classifier on the first turn only.
