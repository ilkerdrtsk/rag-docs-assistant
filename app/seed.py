"""Load the sample documents in `samples/`: `python -m app.seed` (safe to run more than once)."""

import asyncio
from pathlib import Path

from app.config import get_settings
from app.db import SessionLocal, engine
from app.embeddings import build_embedder
from app.ingest import guess_title, ingest_document

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


async def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    async with SessionLocal() as session:
        for path in sorted(SAMPLES.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            document, created = await ingest_document(
                session,
                title=guess_title(text, path.stem),
                text=text,
                source=path.name,
                embedder=embedder,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            state = "added" if created else "already present"
            print(f"{path.name}: {state} ({document.chunk_count} chunks)")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
