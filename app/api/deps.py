import secrets
from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.answering import AnswerGenerator
from app.config import Settings, get_settings
from app.db import get_session
from app.embeddings import Embedder, build_embedder


@lru_cache
def _embedder() -> Embedder | None:
    return build_embedder(get_settings())


def get_embedder() -> Embedder | None:
    return _embedder()


@lru_cache
def _answer_generator() -> AnswerGenerator:
    settings = get_settings()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    # With no key here, the SDK resolves credentials itself (ANTHROPIC_API_KEY, `ant` profile).
    client = anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
    return AnswerGenerator(
        client,
        model=settings.claude_model,
        effort=settings.claude_effort,
        max_tokens=settings.claude_max_tokens,
    )


def get_answer_generator() -> AnswerGenerator:
    return _answer_generator()


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if settings.api_key is None or not settings.api_key.get_secret_value():
        return  # open access, for local development
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key.")


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder | None, Depends(get_embedder)]
GeneratorDep = Annotated[AnswerGenerator, Depends(get_answer_generator)]
