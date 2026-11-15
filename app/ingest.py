"""Adding documents to the knowledge base."""

import hashlib

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import chunk_text
from app.embeddings import Embedder
from app.errors import InvalidDocument
from app.models import Chunk, Document


def guess_title(text: str, fallback: str) -> str:
    """The first Markdown `# Heading` of the text, else the fallback (usually the file name)."""
    for line in text.splitlines():
        if line.strip():
            return line[2:].strip() if line.startswith("# ") and line[2:].strip() else fallback
    return fallback


async def ingest_document(
    session: AsyncSession,
    *,
    title: str,
    text: str,
    source: str | None,
    embedder: Embedder | None,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[Document, bool]:
    """Store a document and its chunks. Returns (document, created).

    Uploading the same content twice returns the existing document instead of
    creating duplicate chunks that would crowd out other results.
    """
    text = text.strip()
    if not text:
        raise InvalidDocument("The document is empty.")
    digest = hashlib.sha256(text.encode()).hexdigest()

    existing = await session.scalar(select(Document).where(Document.content_sha256 == digest))
    if existing is not None:
        return existing, False

    pieces = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
    embeddings: list[list[float] | None] = [None] * len(pieces)
    if embedder is not None:
        embeddings = await embedder.embed([piece.text for piece in pieces], input_type="document")

    document = Document(
        title=title.strip(),
        source=source,
        content_sha256=digest,
        chunk_count=len(pieces),
        chunks=[
            Chunk(position=piece.position, text=piece.text, embedding=embedding)
            for piece, embedding in zip(pieces, embeddings, strict=True)
        ],
    )
    session.add(document)
    try:
        await session.commit()
    except IntegrityError:
        # Someone uploaded the same content at the same moment.
        await session.rollback()
        existing = await session.scalar(select(Document).where(Document.content_sha256 == digest))
        if existing is None:
            raise
        return existing, False
    return document, True


async def embed_missing_chunks(session: AsyncSession, embedder: Embedder, batch: int = 128) -> int:
    """Add embeddings to chunks stored before an embedding provider was configured."""
    total = 0
    while True:
        chunks = list(
            await session.scalars(
                select(Chunk).where(Chunk.embedding.is_(None)).order_by(Chunk.id).limit(batch)
            )
        )
        if not chunks:
            return total
        vectors = await embedder.embed([chunk.text for chunk in chunks], input_type="document")
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        await session.commit()
        total += len(chunks)
