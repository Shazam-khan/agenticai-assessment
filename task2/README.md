# Task 2 — Long-horizon memory & context management

Three explicit memory layers, decay-weighted retrieval, token-budgeted assembly, and a thin intake agent that proves the whole thing works across sessions. The brief flags the trap: stuffing history is overflow; a vector DB alone is noise. The defensible part of this design is the **promotion logic** between layers.

## Architecture

```
                  one user turn
                       v
            +----------------------+
            |   IntakeAgent        |     (task2/intake_agent.py)
            +----------+-----------+
                       |
       +---------------+----------------+
       v                                v
  episodic.store_turn          extractor.extract_facts      (fast LLM)
       |                                |
       v                                v
  +---------------+               +---------------+
  | episodic_turns|               | semantic_facts|<-- chromadb (vectors)
  +---------------+               +-------+-------+
       |                                  ^
       |                                  | upsert on dedupe_key
       |                                  | last_accessed_at refreshes
       |                                  | on every retrieval
       |                                  |
       |     +-----------------+         /
       +---> | working.build   |---------
             |   _context      |        (decay-weighted top-3)
             +-------+---------+
                     |   token budget enforced
                     |   summarise oldest -> working_memory.summary
                     v
            +----------------------+
            |  Reasoning LLM call  |    (Llama 3.3 70B Versatile)
            +----------+-----------+
                       v
              agent reply -> episodic.store_turn
```

The three layers are not just three tables — they're three roles:

- **Episodic** ([shared/memory/episodic.py](../shared/memory/episodic.py)): the raw log. Append-only, every user and agent turn. Used for replay, auditing, and as the source for promotion.
- **Semantic** ([shared/memory/semantic.py](../shared/memory/semantic.py)): structured facts derived from turns. Keyed for dedupe; embedded for retrieval; decay-weighted at read time. The vector store (ChromaDB) carries the embeddings; the SQLite row is the canonical record.
- **Working** ([shared/memory/working.py](../shared/memory/working.py)): the per-session summary + the assembled context for the next LLM call. Token-budget enforced — when the budget would be exceeded, oldest unsummarised raw turns get folded into the rolling summary and marked `summarised=1`.

The **promotion arrows** are the explicit logic the brief asks for:
- *raw turn → fact:* `extractor.extract_facts` runs a fast LLM call per user turn and emits structured `Fact` records. Conservative prompt — extract only what is explicitly stated.
- *fact → retrieval:* `semantic.retrieve_facts` queries the vector store, applies decay weighting, returns top-k above the similarity floor, and bumps `last_accessed_at` + `access_count` on the returned facts (so frequently-recalled facts stay sharp).
- *raw turn → summary:* `working.build_context` triggers `_summarise_oldest` only when the token total exceeds budget. The summariser folds the oldest *unsummarised* turns into the rolling summary and flips their `summarised` flag.

## Decay function

`decay_weight(d_days)` where `d_days = now - last_accessed_at`:

| Days since last access | Weight |
|---|---|
| 0    | 1.00 |
| 90   | 1.00 |
| 180  | 0.50 |
| 270  | 0.25 |
| 365  | ≈0.13 |

Final retrieval score = `cosine_similarity * decay_weight`. Facts are **never deleted** — they just rank below relevant fresh ones. Touching a fact via retrieval resets its clock, so the system has an implicit notion of "still relevant" without manual curation.

Formal definition: `1.0` for `d ≤ 90`, else `0.5 ** ((d - 90) / 90)`. Implemented in [shared/memory/semantic.py:_decay_weight](../shared/memory/semantic.py).

## Token-budget enforcement

`build_context` ([shared/memory/working.py](../shared/memory/working.py)) is the only function the intake agent calls before its reasoning LLM step. It enforces the budget like this:

1. Load `working_memory.summary` (if any), top-3 facts, last 10 unsummarised raw turns.
2. Compute `total_tokens = count_tokens(summary + formatted_facts + raw_turns)`.
3. If `total > budget` (default 4000 via `TOKEN_BUDGET` env var): pick the oldest half of the unsummarised turns, run the summariser (fast LLM), persist the new summary, mark those turns `summarised=1`, recompute. Loop up to 3 times.
4. Return `WorkingContext(summary, facts, raw_turns, token_count, iterations)`.

Test 3 (`test_token_budget_triggers_summarisation`) sets `budget=200` against 8 long turns and asserts the summariser runs and turns get marked.

## Hallucination resistance

Two layers of defence:

1. **Retrieval discipline** (this task): `retrieve_facts` enforces a `similarity_floor` (default 0.25) — low-relevance facts are not injected into the prompt. The hallucination test (`test_agent_does_not_inject_facts_that_were_never_stored`) writes only company + product facts and asks about budget; assertion is that no budget-shaped fact is in the retrieved set.
2. **Model compliance** (Task 4B): the rewritten intake prompt will forbid invention and the eval harness will score the model's adherence. Task 2's job is to prove the retrieval side does not surface phantom facts.

## The 5 tests

`pytest tests/test_task2_memory.py -v` (no API key needed — LLM is stubbed where it appears).

| # | Test | Proves |
|---|---|---|
| 1 | `test_facts_persist_across_sessions` | A fact stored in session A is retrieved in session B for the same `customer_id`. Foreign-customer facts do not leak. |
| 2 | `test_top_k_retrieval_respects_decay` | Two facts with identical embedded text but different ages — fresh one ranks first, day-180 one's `decay_weight` is ~0.5 and `final_score` is strictly lower. Confirms decay is wired into ranking, not just stored on the row. |
| 3 | `test_token_budget_triggers_summarisation` | With `budget=200` and 8 long turns, the summariser runs, `working_memory.summary` is populated, oldest turns get `summarised=1`. |
| 4 | `test_agent_does_not_inject_facts_that_were_never_stored` | The retrieval discipline. Even when asked about a topic that was never discussed, no fact of that category appears in the top-k. |
| 5 | `test_extractor_returns_empty_on_small_talk` | The extractor is conservative — "hi there" yields zero facts. |

## Cross-session demo

`python -m task2.demo` runs two sessions for `lumen_botanicals`:

- **Session 1** (5 turns): Priya introduces Lumen, asks for a hydrating face serum, names a quantity (10k units / 30ml glass bottles), a timeline (Q3 next year / end of June), and a budget (PKR 4M).
- **Session 2** (3 turns): same customer returns. She says "hi", then "did you finalise the timeline?", then "does our budget constraint still work?".

After running, the demo prints every persisted fact for the customer. The interesting evidence is in session 2's per-turn output — for the timeline question the agent recalls `[timeline] end of June` + `[timeline] Q3 next year`; for the budget question it recalls `[budget] PKR 4 million`. None of these were re-stated in session 2.

## Run it

```powershell
# Unit tests (fast, no API key)
.venv\Scripts\python.exe -m pytest tests/test_task2_memory.py -v

# Cross-session demo (real Groq; ~30s)
.venv\Scripts\python.exe -m task2.demo

# Inspect what's persisted
.venv\Scripts\python.exe -c "from shared.db import get_conn; c=get_conn(); print('\\nFACTS:'); [print(dict(r)) for r in c.execute('SELECT customer_id, category, entity, value, access_count FROM semantic_facts ORDER BY customer_id, category')]"
```

## Tradeoffs (specific to Task 2)

| Choice | Win | Lose / upgrade path |
|---|---|---|
| **Exact-text dedupe** on `sha256(customer + category + entity + value_normalised)` | Deterministic, zero LLM cost on every write, easy to audit. | "10k units" and "10,000 units" become two rows. Upgrade: normaliser pass that canonicalises numbers + units before hashing, or a nightly semantic-dedupe job that merges near-duplicates. |
| **Token count = `len // 4`** (not a real tokenizer) | Zero extra dependency on a Groq/OpenAI tokenizer; deterministic; fast. | ~10% inaccuracy → summarise slightly earlier or later than the true budget. Upgrade: swap to `tiktoken` cl100k_base — close enough for Llama, one-line change in [shared/memory/working.py:count_tokens](../shared/memory/working.py). |
| **Live extraction per turn** (one fast-LLM call before the reasoning call) | Cross-session demo works without an end-of-session signal; facts are always available for retrieval; failure of extraction never blocks the agent's reply. | Doubles LLM call count per turn. Upgrade: extraction can be async/background — the reasoning call doesn't depend on the new fact being indexed if it's already in episodic. |
| **Local sentence-transformers `all-MiniLM-L6-v2`** (80MB, ~1.5s cold-start, 384-dim) | No embedding API key required; free; deterministic. | Weaker than `text-embedding-3-large` on nuanced similarity; first call is slow. Upgrade: swap implementation in [shared/embeddings.py](../shared/embeddings.py) (one function) or warm-load on import. |
| **Monotonic decay** with a hard 90-day shelf | Debuggable, no eval-leakage risk, no learned-decay test surface. | A fact a customer didn't *mention* for 90 days may still be relevant. Upgrade: per-category decay rates (contact info decays slowly, quantities/timelines decay fast). |
| **Conservative extractor prompt** ("only what is explicitly stated") | High precision; few hallucinated facts; cheap. | Sometimes mis-categorises ("10,000 units" went to `product_need` once in the demo where `quantity` would have been better). Upgrade: tighten the category descriptions or add a 2-shot example per category in the prompt. |
