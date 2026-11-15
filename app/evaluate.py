"""Measure retrieval quality on a labelled question set: `python -m app.evaluate`.

For each question the expected document must appear among the top-k retrieved
chunks. Reports hit rate and mean reciprocal rank (MRR), so changes to chunking or
search can be compared with numbers instead of impressions. It does not call Claude.
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import Embedder
from app.retrieval import hybrid_search

DEFAULT_EVAL_FILE = Path(__file__).resolve().parent.parent / "evals" / "retrieval.jsonl"


@dataclass(frozen=True)
class EvalResult:
    question: str
    expected: str
    rank: int | None  # 1-based rank of the first chunk from the expected document


@dataclass(frozen=True)
class EvalReport:
    results: list[EvalResult]
    k: int

    @property
    def hit_rate(self) -> float:
        return sum(result.rank is not None for result in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        return sum(1 / result.rank for result in self.results if result.rank) / len(self.results)


def load_cases(path: Path = DEFAULT_EVAL_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


async def evaluate_retrieval(
    session: AsyncSession, cases: list[dict], *, k: int, embedder: Embedder | None
) -> EvalReport:
    results = []
    for case in cases:
        chunks = await hybrid_search(session, case["question"], top_k=k, embedder=embedder)
        titles = [chunk.document_title for chunk in chunks]
        rank = titles.index(case["expected"]) + 1 if case["expected"] in titles else None
        results.append(EvalResult(case["question"], case["expected"], rank))
    return EvalReport(results, k)


async def main(k: int) -> None:  # pragma: no cover - thin CLI wrapper
    from app.config import get_settings
    from app.db import SessionLocal, engine
    from app.embeddings import build_embedder

    async with SessionLocal() as session:
        report = await evaluate_retrieval(
            session, load_cases(), k=k, embedder=build_embedder(get_settings())
        )
    await engine.dispose()
    for result in report.results:
        status = f"rank {result.rank}" if result.rank else "MISS"
        print(f"{status:>7}  {result.question}  ->  {result.expected}")
    count = len(report.results)
    print(f"\nhit@{k}: {report.hit_rate:.0%}   MRR: {report.mrr:.2f}   ({count} questions)")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="How many chunks to retrieve")
    asyncio.run(main(parser.parse_args().k))
