"""Test setup: a separate `<db>_test` database, a fake Claude client and a fake embedder."""

import asyncio
import hashlib
import math
import os
import re

import asyncpg
import pytest
from sqlalchemy.engine import make_url

from app.config import Settings

# Point the app at a dedicated test database *before* app.db creates its engine.
_base_url = make_url(Settings().database_url)
TEST_DATABASE_URL = _base_url.set(database=f"{_base_url.database}_test")
os.environ["DATABASE_URL"] = TEST_DATABASE_URL.render_as_string(hide_password=False)


async def _create_test_database() -> None:
    conn = await asyncpg.connect(
        user=_base_url.username,
        password=_base_url.password,
        host=_base_url.host,
        port=_base_url.port,
        database=_base_url.database,
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DATABASE_URL.database
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{TEST_DATABASE_URL.database}"')
    finally:
        await conn.close()


asyncio.run(_create_test_database())

from anthropic.types.beta import BetaMessage  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.answering import AnswerGenerator  # noqa: E402
from app.api.deps import get_answer_generator, get_embedder  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.ingest import guess_title, ingest_document  # noqa: E402
from app.main import app  # noqa: E402
from app.models import EMBEDDING_DIM, Base  # noqa: E402
from app.seed import SAMPLES  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
async def schema():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_tables(schema):
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))


@pytest.fixture
async def session():
    async with SessionLocal() as db_session:
        yield db_session


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words end up close together."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def embed(self, texts, *, input_type):
        self.calls.append((input_type, len(texts)))
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text: str) -> list[float]:
        values = [0.0] * EMBEDDING_DIM
        values[0] = 0.01  # never an all-zero vector
        for word in re.findall(r"[a-z]+", text.lower()):
            values[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


def claude_reply(*blocks: dict, stop_reason: str = "end_turn") -> BetaMessage:
    """A response object exactly as the Anthropic SDK returns it."""
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 80},
        }
    )


def cited(text_: str, *, document_index: int, cited_text: str) -> dict:
    return {
        "type": "text",
        "text": text_,
        "citations": [
            {
                "type": "char_location",
                "cited_text": cited_text,
                "document_index": document_index,
                "document_title": "ignored",
                "start_char_index": 0,
                "end_char_index": len(cited_text),
                "file_id": None,
            }
        ],
    }


class FakeClaude:
    """Stands in for `anthropic.AsyncAnthropic`; records requests and returns a set reply."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply = claude_reply({"type": "text", "text": "No reply configured."})
        self.error: Exception | None = None
        self.beta = self
        self.messages = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def claude() -> FakeClaude:
    return FakeClaude()


@pytest.fixture
def generator(claude) -> AnswerGenerator:
    return AnswerGenerator(claude, model="claude-opus-5-5", effort="medium", max_tokens=16000)


@pytest.fixture
def settings() -> Settings:
    return Settings(top_k=4, api_key=None, anthropic_api_key=None, voyage_api_key=None)


@pytest.fixture
def embedder() -> FakeEmbedder | None:
    """Keyword-only by default; tests that need vectors override this fixture."""
    return None


@pytest.fixture
async def client(settings, generator, embedder):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_answer_generator] = lambda: generator
    app.dependency_overrides[get_embedder] = lambda: embedder
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture
async def samples(session, embedder):
    """The three sample documents from `samples/`, ingested like `python -m app.seed` does."""
    for path in sorted(SAMPLES.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        await ingest_document(
            session,
            title=guess_title(content, path.stem),
            text=content,
            source=path.name,
            embedder=embedder,
            chunk_size=1200,
            chunk_overlap=200,
        )
