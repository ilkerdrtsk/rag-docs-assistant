"""Embed chunks stored before VOYAGE_API_KEY was set: `python -m app.reindex`."""

import asyncio

from app.config import get_settings
from app.db import SessionLocal, engine
from app.embeddings import build_embedder
from app.ingest import embed_missing_chunks


async def main() -> None:
    embedder = build_embedder(get_settings())
    if embedder is None:
        raise SystemExit("Set VOYAGE_API_KEY first; there is nothing to embed with.")
    async with SessionLocal() as session:
        count = await embed_missing_chunks(session, embedder)
    await engine.dispose()
    print(f"Embedded {count} chunk(s).")


if __name__ == "__main__":
    asyncio.run(main())
