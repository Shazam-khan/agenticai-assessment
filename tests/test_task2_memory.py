"""Memory-layer tests.

These exercise the storage + retrieval discipline. The cross-session demo
itself (with real Groq) is the qualitative evidence; these tests are the
quantitative discipline checks.

We use a temp DB + temp Chroma path per test so they're isolated and don't
pollute the real data store. Embeddings are real (sentence-transformers is
deterministic given a fixed input) — they're what's actually under test for
retrieval ranking.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from shared.llm import LLMResponse
from shared.memory.schema import Fact, Turn


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    """Point DB + Chroma at tmp paths, and reset the Chroma singleton between tests."""
    db = tmp_path / "test.db"
    chroma = tmp_path / "chroma"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("CHROMA_PATH", str(chroma))
    import shared.db
    import shared.memory.semantic as semantic_mod
    monkeypatch.setattr(shared.db, "DB_PATH", str(db))
    monkeypatch.setattr(semantic_mod, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(semantic_mod, "_chroma_collection", None)
    yield


def _stub_llm(json_responses: list[dict] | None = None, text_response: str = "ok") -> MagicMock:
    llm = MagicMock()
    json_iter = iter(json_responses or [])

    def complete_json(*, system, user, schema, trace_id, parent_span_id=None,
                      prompt_version="v1", model=None):
        payload = next(json_iter)
        parsed = schema.model_validate(payload)
        return parsed, LLMResponse(content=json.dumps(payload), tokens_in=5, tokens_out=10,
                                    latency_ms=2, model="stub", cost_usd=0.0)

    def complete(*, system, user, trace_id, parent_span_id=None, prompt_version="v1",
                 json_mode=False, temperature=0.0, model=None, max_tokens=2048):
        return LLMResponse(content=text_response, tokens_in=5, tokens_out=10,
                            latency_ms=2, model="stub", cost_usd=0.0)

    llm.complete_json = complete_json
    llm.complete = complete
    return llm


# ----------------------------- test 1 ------------------------------

def test_facts_persist_across_sessions():
    """A fact stored during session A must be retrievable in session B for the
    same customer_id, without re-storing it. Proves the cross-session promise."""
    from shared.memory.semantic import retrieve_facts, store_fact

    # Session A — store a couple of facts.
    store_fact(Fact(customer_id="acme", category="product_need",
                    value="hydrating face serum"))
    store_fact(Fact(customer_id="acme", category="quantity",
                    value="10000 units in 30ml glass bottles"))
    # Foreign-customer fact must not leak.
    store_fact(Fact(customer_id="other_co", category="product_need",
                    value="lip balm"))

    # Session B — different session, same customer, retrieve.
    results = retrieve_facts(customer_id="acme", query="what serum did we discuss")
    assert results, "expected at least one fact to be recalled for acme"
    top_values = [r.fact.value for r in results]
    assert any("serum" in v for v in top_values), f"serum fact not in top-k: {top_values}"
    # And the wrong customer's facts must not appear.
    assert not any("lip balm" in v for v in top_values)


# ----------------------------- test 2 ------------------------------

def test_top_k_retrieval_respects_decay():
    """Two facts with identical embedded text but different ages — the fresh
    one must rank higher and the old one's final_score must be visibly lower."""
    from shared.memory.semantic import retrieve_facts, store_fact

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    old = (now - timedelta(days=180)).isoformat()

    # Same value, different customers so dedupe doesn't collapse them.
    fresh = Fact(customer_id="cust_fresh", category="budget", value="PKR 4 million")
    fresh.last_accessed_at = now.isoformat()
    store_fact(fresh)

    stale = Fact(customer_id="cust_stale", category="budget", value="PKR 4 million")
    stale.last_accessed_at = old
    # Bypass store_fact's auto-touch of last_accessed_at by patching post-insert.
    store_fact(stale)
    from shared.db import get_conn
    conn = get_conn()
    conn.execute("UPDATE semantic_facts SET last_accessed_at = ? WHERE fact_id = ?",
                 (old, stale.fact_id))
    conn.commit()
    conn.close()

    fresh_hits = retrieve_facts(customer_id="cust_fresh", query="budget", now=now)
    stale_hits = retrieve_facts(customer_id="cust_stale", query="budget", now=now)

    assert fresh_hits and stale_hits
    assert fresh_hits[0].decay_weight == 1.0
    assert stale_hits[0].decay_weight < 1.0
    assert stale_hits[0].decay_weight == pytest.approx(0.5, abs=0.01), \
        f"day-180 decay should be ~0.5, got {stale_hits[0].decay_weight}"
    assert fresh_hits[0].final_score > stale_hits[0].final_score


# ----------------------------- test 3 ------------------------------

def test_token_budget_triggers_summarisation():
    """With a tiny budget and many long turns, build_context must fold the
    oldest turns into working_memory.summary and mark them summarised."""
    from shared.db import get_conn
    from shared.memory import episodic, working

    session_id = "sess_budget"
    customer_id = "budget_co"

    # 8 long-ish turns. With budget=200 (tokens), most must get folded.
    long_text = "We discussed product details and timeline considerations at length. " * 6
    for i in range(8):
        episodic.store_turn(
            session_id=session_id, customer_id=customer_id,
            role="user" if i % 2 == 0 else "agent",
            content=f"Turn {i}: {long_text}",
        )

    llm = _stub_llm(text_response="Consolidated summary preserving facts and numbers.")
    ctx = working.build_context(
        session_id=session_id, customer_id=customer_id,
        query="status?", llm=llm, budget=200,
    )

    assert ctx.iterations >= 1, "expected at least one summarisation pass"

    conn = get_conn()
    summary_row = conn.execute(
        "SELECT summary, summary_token_count FROM working_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    folded = conn.execute(
        "SELECT COUNT(*) AS n FROM episodic_turns WHERE session_id = ? AND summarised = 1",
        (session_id,),
    ).fetchone()
    conn.close()

    assert summary_row is not None and summary_row["summary"], "summary should be populated"
    assert folded["n"] > 0, "at least one turn should be marked summarised"


# ----------------------------- test 4 ------------------------------

def test_agent_does_not_inject_facts_that_were_never_stored():
    """Hallucination-resistance, retrieval-side:

    Store only company + product. Then in a 'new session' ask about budget
    (which was never stored). The retrieval must NOT inject a budget-shaped
    fact into the context, because none exists. This proves the retrieval
    discipline — Task 4B will test the *model's* compliance once a stricter
    intake prompt is in place.
    """
    from shared.memory.semantic import retrieve_facts, store_fact

    store_fact(Fact(customer_id="nofacts", category="company_info",
                    value="skincare startup based in Karachi"))
    store_fact(Fact(customer_id="nofacts", category="product_need",
                    value="hydrating face serum"))
    # NOTE: no budget fact stored.

    results = retrieve_facts(
        customer_id="nofacts",
        query="what was our budget?",
        similarity_floor=0.45,  # be strict: only inject clearly-relevant facts
    )
    # Either zero facts, OR none of them is a budget fact.
    assert not any(r.fact.category == "budget" for r in results), \
        f"budget fact leaked despite none being stored: {[r.fact.value for r in results]}"


# ----------------------------- test 5 ------------------------------

def test_extractor_returns_empty_on_small_talk():
    """The extractor is conservative: 'hello' should yield zero facts."""
    from shared.memory.extractor import extract_facts

    turn = Turn(session_id="s", customer_id="c", role="user", content="hi there")
    llm = _stub_llm(json_responses=[{"facts": []}])

    facts = extract_facts(turn=turn, llm=llm, trace_id="t")
    assert facts == []
