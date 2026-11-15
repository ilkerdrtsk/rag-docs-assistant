"""Optional embedding provider for semantic search (Voyage AI)."""

from itertools import batched
from typing import Literal, Protocol

import httpx

from app.config import Settings
from app.errors import EmbeddingError
from app.models import EMBEDDING_DIM


class Embedder(Protocol):
    async def embed(
        self, texts: list[str], *, input_type: Literal["document", "query"]
    ) -> list[list[float]]: ...


class VoyageEmbedder:
    URL = "https://api.voyageai.com/v1/embeddings"
    BATCH_SIZE = 128

    def __init__(self, api_key: str, model: str, *, client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=30)

    async def embed(
        self, texts: list[str], *, input_type: Literal["document", "query"]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(texts, self.BATCH_SIZE):
            try:
                response = await self._client.post(
                    self.URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "input": list(batch),
                        "model": self._model,
                        # Voyage tunes the vector for the side of the search it is on.
                        "input_type": input_type,
                        "output_dimension": EMBEDDING_DIM,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    "The embedding provider could not process the request."
                ) from exc
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in items)
        return vectors


def build_embedder(settings: Settings) -> Embedder | None:
    if settings.voyage_api_key is None or not settings.voyage_api_key.get_secret_value():
        return None
    return VoyageEmbedder(settings.voyage_api_key.get_secret_value(), settings.voyage_model)
