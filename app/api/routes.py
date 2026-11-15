import uuid
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from sqlalchemy import select

from app import retrieval
from app.api.deps import EmbedderDep, GeneratorDep, SessionDep, SettingsDep, require_api_key
from app.errors import InvalidDocument, NotFound
from app.ingest import guess_title, ingest_document
from app.models import Document
from app.schemas import (
    AnswerOut,
    AskIn,
    ChunkOut,
    DocumentIn,
    DocumentOut,
    ErrorOut,
    IngestOut,
    SearchIn,
    UsageOut,
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}
ERRORS = {code: {"model": ErrorOut} for code in (401, 404, 422, 429, 502)}

router = APIRouter(dependencies=[Depends(require_api_key)], responses=ERRORS)


async def _ingest(session, settings, embedder, *, title, text, source, response: Response):
    document, created = await ingest_document(
        session,
        title=title,
        text=text,
        source=source,
        embedder=embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return IngestOut(document=DocumentOut.model_validate(document), created=created)


@router.post("/documents", response_model=IngestOut, status_code=201, tags=["documents"])
async def add_document(
    payload: DocumentIn,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
):
    """Add a document from JSON. Identical content is stored only once (200 instead of 201)."""
    return await _ingest(
        session,
        settings,
        embedder,
        title=payload.title,
        text=payload.text,
        source=payload.source,
        response=response,
    )


@router.post("/documents/upload", response_model=IngestOut, status_code=201, tags=["documents"])
async def upload_document(
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
    file: Annotated[UploadFile, File(description="A .txt or .md file, UTF-8, up to 2 MB")],
    title: Annotated[str | None, Form(max_length=200)] = None,
):
    filename = PurePath(file.filename or "upload.txt").name
    if PurePath(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise InvalidDocument("Only .txt and .md files are supported.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidDocument("The file is larger than 2 MB.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidDocument("The file must be UTF-8 text.") from exc
    return await _ingest(
        session,
        settings,
        embedder,
        title=title or guess_title(text, PurePath(filename).stem),
        text=text,
        source=filename,
        response=response,
    )


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
async def list_documents(session: SessionDep):
    return await session.scalars(select(Document).order_by(Document.created_at, Document.id))


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: uuid.UUID, session: SessionDep):
    document = await session.get(Document, document_id)
    if document is None:
        raise NotFound("Document not found.")
    await session.delete(document)
    await session.commit()


@router.post("/search", response_model=list[ChunkOut], tags=["question answering"])
async def search(
    payload: SearchIn, session: SessionDep, settings: SettingsDep, embedder: EmbedderDep
):
    """Retrieval only, without calling Claude. Useful to see why an answer was (not) found."""
    return await retrieval.hybrid_search(
        session, payload.query, top_k=payload.top_k or settings.top_k, embedder=embedder
    )


@router.post("/ask", response_model=AnswerOut, tags=["question answering"])
async def ask(
    payload: AskIn,
    session: SessionDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
    generator: GeneratorDep,
):
    """Answer a question from the knowledge base, with numbered citations."""
    chunks = await retrieval.hybrid_search(
        session, payload.question, top_k=payload.top_k or settings.top_k, embedder=embedder
    )
    answer = await generator.answer(payload.question, chunks)
    return AnswerOut(
        answer=answer.text,
        found=answer.found,
        citations=answer.citations,
        sources=answer.sources,
        model=answer.model,
        usage=UsageOut(input_tokens=answer.input_tokens, output_tokens=answer.output_tokens),
    )


health_router = APIRouter()


@health_router.get("/health", tags=["ops"])
async def health(session: SessionDep, embedder: EmbedderDep):
    await session.execute(select(1))
    return {"status": "ok", "search": "hybrid" if embedder else "keyword"}
