"""Semantic memory — structured facts with decay-weighted retrieval.

Three things make this more than 'a vector DB':
  1. **Explicit promotion** from episodic — facts come from the extractor, not raw turns.
  2. **Dedupe** on (customer, category, entity, value_norm) — re-asserted facts
     update access_count + last_accessed_at, they don't create a new row.
  3. **Decay-weighted ranking** — facts not accessed in 90+ days drop in score
     but are never deleted. last_accessed_at refreshes on every retrieval, so
     facts the agent uses stay sharp; ignored facts fade.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from ..db import get_conn
from ..embeddings import embed_one
from ..trace import trace
from .schema import Fact, RetrievedFact, _now_iso

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = "semantic_facts"

_chroma_collection = None


def _collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


# ---- decay ------------------------------------------------------------------

def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _decay_weight(last_accessed_at: str, now: datetime | None = None) -> float:
    """1.0 for the first 90 days, then exponential half-life of 90 days.

    day 0   -> 1.0
    day 90  -> 1.0
    day 180 -> 0.5
    day 270 -> 0.25
    day 365 -> ~0.13
    """
    now = now or datetime.now(timezone.utc)
    age_days = (now - _parse_iso(last_accessed_at)).total_seconds() / 86400.0
    if age_days <= 90:
        return 1.0
    return 0.5 ** ((age_days - 90) / 90)


# ---- store + upsert ---------------------------------------------------------

def store_fact(
    fact: Fact,
    *,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> Fact:
    """Upsert on dedupe_key. Existing row: bump access_count, update last_accessed,
    keep older created_at and original fact_id. Embeds + writes to Chroma."""
    fact = fact.with_dedupe_key()
    with trace(
        "memory.semantic.store_fact",
        "tool",
        trace_id=trace_id or fact.customer_id,
        parent_span_id=parent_span_id,
        input_payload={"category": fact.category, "entity": fact.entity, "value": fact.value[:100]},
    ) as rec:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT fact_id, access_count FROM semantic_facts WHERE dedupe_key = ?",
                (fact.dedupe_key,),
            ).fetchone()
            if row:
                # Upsert path — preserve original fact_id, bump counters.
                fact.fact_id = row["fact_id"]
                conn.execute(
                    """UPDATE semantic_facts
                       SET access_count = access_count + 1,
                           last_accessed_at = ?
                       WHERE fact_id = ?""",
                    (_now_iso(), fact.fact_id),
                )
                conn.commit()
                rec.output_payload = {"upserted": True, "fact_id": fact.fact_id}
                return fact

            conn.execute(
                """INSERT INTO semantic_facts
                   (fact_id, customer_id, category, entity, value, confidence,
                    source_turn_id, created_at, last_accessed_at, access_count, dedupe_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fact.fact_id, fact.customer_id, fact.category, fact.entity, fact.value,
                 fact.confidence, fact.source_turn_id, fact.created_at, fact.last_accessed_at,
                 fact.access_count, fact.dedupe_key),
            )
            conn.commit()
        finally:
            conn.close()

        # Vector side. Document text is what the retriever queries against.
        doc = f"{fact.category}: {fact.value}"
        if fact.entity:
            doc = f"{fact.category} for {fact.entity}: {fact.value}"
        _collection().upsert(
            ids=[fact.fact_id],
            embeddings=[embed_one(doc)],
            documents=[doc],
            metadatas=[{"customer_id": fact.customer_id, "category": fact.category}],
        )
        rec.output_payload = {"inserted": True, "fact_id": fact.fact_id}
    return fact


# ---- retrieve ---------------------------------------------------------------

def _load_fact(fact_id: str) -> Fact | None:
    conn = get_conn()
    try:
        row = conn.execute(
            """SELECT fact_id, customer_id, category, entity, value, confidence,
                      source_turn_id, created_at, last_accessed_at, access_count, dedupe_key
               FROM semantic_facts WHERE fact_id = ?""",
            (fact_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return Fact(**dict(row))


def _touch_facts(fact_ids: Iterable[str]) -> None:
    ids = list(fact_ids)
    if not ids:
        return
    placeholders = ",".join(["?"] * len(ids))
    conn = get_conn()
    try:
        conn.execute(
            f"""UPDATE semantic_facts
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE fact_id IN ({placeholders})""",
            [_now_iso(), *ids],
        )
        conn.commit()
    finally:
        conn.close()


def retrieve_facts(
    *,
    customer_id: str,
    query: str,
    top_k: int = 3,
    similarity_floor: float = 0.25,
    candidate_pool: int = 15,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    now: datetime | None = None,
) -> list[RetrievedFact]:
    with trace(
        "memory.semantic.retrieve",
        "tool",
        trace_id=trace_id or customer_id,
        parent_span_id=parent_span_id,
        input_payload={"customer_id": customer_id, "query": query[:200], "top_k": top_k},
    ) as rec:
        col = _collection()
        # Cap pool by what exists for this customer (avoids Chroma warning).
        results = col.query(
            query_embeddings=[embed_one(query)],
            n_results=candidate_pool,
            where={"customer_id": customer_id},
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if not ids:
            rec.output_payload = {"returned": 0}
            return []

        # Chroma cosine distance => similarity = 1 - distance (already on cosine space).
        ranked: list[RetrievedFact] = []
        for fact_id, dist in zip(ids, distances):
            fact = _load_fact(fact_id)
            if not fact:
                continue
            similarity = max(0.0, 1.0 - float(dist))
            decay = _decay_weight(fact.last_accessed_at, now=now)
            final = similarity * decay
            if final < similarity_floor:
                continue
            ranked.append(RetrievedFact(
                fact=fact, similarity=similarity, decay_weight=decay, final_score=final,
            ))

        ranked.sort(key=lambda r: r.final_score, reverse=True)
        top = ranked[:top_k]

        # Promotion / reinforcement: touched facts stay sharp.
        _touch_facts([r.fact.fact_id for r in top])
        rec.output_payload = {
            "returned": len(top),
            "fact_ids": [r.fact.fact_id for r in top],
        }
        return top
