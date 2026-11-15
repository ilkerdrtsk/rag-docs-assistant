import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)
    source: str | None = Field(default=None, max_length=255)


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source: str | None
    chunk_count: int
    created_at: datetime


class IngestOut(BaseModel):
    document: DocumentOut
    created: bool = Field(description="False when identical content was already stored")


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: int
    document_id: uuid.UUID
    document_title: str
    position: int
    text: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    number: int
    chunk_id: int
    document_title: str
    cited_text: str


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class AnswerOut(BaseModel):
    answer: str
    found: bool = Field(description="False when the knowledge base had nothing to cite")
    citations: list[CitationOut]
    sources: list[ChunkOut] = Field(description="Cited chunks, in marker order ([1], [2], ...)")
    model: str | None
    usage: UsageOut


class ErrorOut(BaseModel):
    code: str
    detail: str
