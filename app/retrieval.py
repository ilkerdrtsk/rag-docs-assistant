"""Hybrid retrieval: PostgreSQL full-text search plus optional pgvector similarity.

Keyword search is good at exact terms (product names, error codes); vector search is
good at paraphrases ("send it back" vs "return"). Their rankings are merged with
Reciprocal Rank Fusion, which needs no score calibration between the two.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import Embedder
from app.models import Chunk, Document

RRF_K = 60  # standard constant from the original RRF paper


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: uuid.UUID
    document_title: str
    position: int
    text: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None


def _any_word_query(query: str):
    """`plainto_tsquery` joins words with AND; questions match better with OR plus ranking."""
    and_query = cast(func.plainto_tsquery("english", query), Text)
    return func.to_tsquery("english", func.replace(and_query, "&", "|"))


async def keyword_search(session: AsyncSession, query: str, limit: int) -> list[int]:
    tsquery = _any_word_query(query)
    rank = func.ts_rank_cd(Chunk.search_vector, tsquery)
    rows = await session.scalars(
        select(Chunk.id)
        .where(Chunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return list(rows)


async def vector_search(session: AsyncSession, embedding: list[float], limit: int) -> list[int]:
    rows = await session.scalars(
        select(Chunk.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(embedding), Chunk.id)
        .limit(limit)
    )
    return list(rows)


def reciprocal_rank_fusion(*rankings: list[int], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


async def hybrid_search(
    session: AsyncSession, query: str, *, top_k: int, embedder: Embedder | None
) -> list[RetrievedChunk]:
    candidates = top_k * 4
    keyword_ids = await keyword_search(session, query, candidates)
    vector_ids: list[int] = []
    if embedder is not None:
        [query_embedding] = await embedder.embed([query], input_type="query")
        vector_ids = await vector_search(session, query_embedding, candidates)

    scores = reciprocal_rank_fusion(keyword_ids, vector_ids)
    best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    if not best:
        return []

    rows = await session.execute(
        select(Chunk, Document.title).join(Document).where(Chunk.id.in_(best))
    )
    by_id = {chunk.id: (chunk, title) for chunk, title in rows}
    keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=by_id[chunk_id][0].document_id,
            document_title=by_id[chunk_id][1],
            position=by_id[chunk_id][0].position,
            text=by_id[chunk_id][0].text,
            score=round(scores[chunk_id], 6),
            keyword_rank=keyword_rank.get(chunk_id),
            vector_rank=vector_rank.get(chunk_id),
        )
        for chunk_id in best
    ]
