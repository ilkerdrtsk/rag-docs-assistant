import pytest

from app.evaluate import evaluate_retrieval, load_cases
from app.ingest import embed_missing_chunks, ingest_document
from app.models import Chunk
from app.retrieval import hybrid_search, reciprocal_rank_fusion
from tests.conftest import FakeEmbedder


def test_rrf_rewards_chunks_found_by_both_searches():
    scores = reciprocal_rank_fusion([1, 2, 3], [3, 4])

    ranked = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
    assert ranked[0] == 3  # 3rd in one list and 1st in the other beats 1st in only one
    assert set(scores) == {1, 2, 3, 4}


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([], []) == {}


async def add(session, title, text, embedder=None):
    document, _ = await ingest_document(
        session,
        title=title,
        text=text,
        source=None,
        embedder=embedder,
        chunk_size=1200,
        chunk_overlap=200,
    )
    return document


async def test_keyword_search_finds_the_right_document(session, samples):
    results = await hybrid_search(
        session, "How much does express shipping cost?", top_k=3, embedder=None
    )

    assert results[0].document_title == "Shipping"
    assert results[0].keyword_rank == 1
    assert results[0].vector_rank is None


async def test_words_are_matched_with_or_not_and(session):
    await add(session, "Returns", "Returns are accepted within 45 days.")

    # "unicorn" appears nowhere; an AND query would return nothing.
    results = await hybrid_search(session, "returns unicorn", top_k=3, embedder=None)

    assert [result.document_title for result in results] == ["Returns"]


async def test_no_match_returns_nothing(session, samples):
    assert await hybrid_search(session, "quantum chromodynamics", top_k=3, embedder=None) == []


@pytest.mark.parametrize("embedder", [FakeEmbedder()], indirect=False)
async def test_vector_search_finds_what_keyword_search_cannot(session, embedder):
    # "whom" is an English stop word, so full-text search ignores it entirely.
    await add(session, "Contacts", "Whom to call: the duty manager.", embedder)
    await add(session, "Other", "Completely unrelated paragraph.", embedder)

    results = await hybrid_search(session, "whom", top_k=1, embedder=embedder)

    assert results[0].document_title == "Contacts"
    assert results[0].keyword_rank is None
    assert results[0].vector_rank == 1
    assert ("query", 1) in embedder.calls


async def test_hybrid_results_carry_both_ranks(session):
    embedder = FakeEmbedder()
    await add(session, "Shipping", "Express shipping costs 12 EUR.", embedder)

    [result] = await hybrid_search(session, "express shipping cost", top_k=3, embedder=embedder)

    assert (result.keyword_rank, result.vector_rank) == (1, 1)
    assert result.score == pytest.approx(2 / 61, abs=1e-6)


async def test_missing_embeddings_can_be_added_later(session):
    await add(session, "Old", "Stored before embeddings were configured.")
    embedder = FakeEmbedder()

    assert await embed_missing_chunks(session, embedder) == 1
    chunk = await session.get(Chunk, 1)
    assert chunk.embedding is not None


async def test_sample_retrieval_eval_scores_full_marks(session, samples):
    report = await evaluate_retrieval(session, load_cases(), k=3, embedder=None)

    misses = [result.question for result in report.results if result.rank is None]
    assert misses == []
    assert report.hit_rate == 1.0
    assert report.mrr >= 0.9
