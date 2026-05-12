# Agentic AI Engineer — Assessment

Four interconnected tasks, one system. See [ARCHITECTURE.md](ARCHITECTURE.md) for the shared primitives (message envelope, trace table, memory layers, HITL pause/resume, circuit breaker, eval runner, observability surface). Each task's README below explains the slice it owns.

| Task | Folder | Status |
|------|--------|--------|
| 1 — Multi-agent orchestration | [task1/](task1/) | done — 5/5 eval + 3/3 unit |
| 2 — Memory architecture | [task2/](task2/) | done — 5/5 unit + cross-session demo |
| 3 — Tools, HITL, circuit breaker | [task3/](task3/) | done — 30/30 unit + HITL pause/resume demo + Task 1 still 5/5 |
| 4 — Prompts, evals, observability | [task4/](task4/) | done — 9/9 unit + eval harness + Flask dashboard |

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env       # fill in GROQ_API_KEY
```

## What I would build next

Concrete next steps, in priority order — none of these are speculative:

1. **Move eval to the 70B model.** The 8B model the harness currently runs on misses some legitimate multi-field-extraction cases (the F3 timeline failure is documented in [task4/README.md](task4/README.md)). The production agent already runs on 70B; aligning the eval would close the floor/ceiling gap. One env var flip + a paid Groq tier.
2. **Per-(tool, key_arg) circuit breakers.** Right now one bad supplier opens the circuit for all `create_purchase_order` calls, not just orders to that supplier. Documented as tradeoff #13 in [ARCHITECTURE.md](ARCHITECTURE.md). Schema change: extend `circuit_state.tool_name` to `(tool_name, scope_key)`.
3. **Surgical resume for the HITL pause/resume flow.** Today, `supervisor.resume()` replays the entire plan; idempotency keys absorb the duplication but we pay for every LLM call twice. A `span_outputs` cache keyed by `(trace_id, span_path)` would skip already-traced work. Tradeoff #11.
4. **Async fact extraction during intake.** Tradeoff #10: live per-turn extraction doubles the LLM call count per turn. Moving the extraction to a background queue would halve user-perceived latency. The cross-session demo still works because facts are available for retrieval by the NEXT turn.
5. **Tokenizer-accurate budget enforcement.** The `count_tokens = len(text) // 4` approximation in [shared/memory/working.py](shared/memory/working.py) is good enough for the budget trip but drifts by ~10%. A one-line swap to `tiktoken` cl100k_base closes that. Tradeoff #9.
6. **Notification adapter for `flag_bottleneck`.** Today the tool appends to `data/notifications.log`. The adapter point is one function in [shared/tools/production_tools.py](shared/tools/production_tools.py); production routes critical-severity alerts to PagerDuty and others to Slack.
7. **Routing-classifier eval suite.** Task 1's planner over-routing was caught once by the existing eval; documented as tradeoff #7. A nightly suite that scores routing precision on every prompt change + model bump would catch regressions before they ship.

The brief asks for "a 1-2 page architecture document explaining how the four tasks connect as a single system." That's [ARCHITECTURE.md](ARCHITECTURE.md) — single message envelope, single trace table, single eval runner, three-layer memory promotion, persist-and-exit HITL — and 17 explicit tradeoffs with upgrade paths.
