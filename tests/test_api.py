import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select

from app.embeddings import VoyageEmbedder
from app.errors import EmbeddingError
from app.models import Chunk
from tests.conftest import FakeEmbedder, cited, claude_reply


class TestDocuments:
    async def test_add_list_and_delete(self, client):
        created = await client.post(
            "/documents", json={"title": "Policy", "text": "Returns within 45 days."}
        )
        assert created.status_code == 201
        body = created.json()
        assert body["created"] is True
        assert body["document"]["chunk_count"] == 1

        listing = await client.get("/documents")
        assert [doc["title"] for doc in listing.json()] == ["Policy"]

        doc_id = body["document"]["id"]
        assert (await client.delete(f"/documents/{doc_id}")).status_code == 204
        assert (await client.delete(f"/documents/{doc_id}")).status_code == 404

    async def test_identical_content_is_stored_once(self, client, session):
        payload = {"title": "Policy", "text": "Returns within 45 days."}
        await client.post("/documents", json=payload)

        again = await client.post("/documents", json={**payload, "title": "Copy"})

        assert again.status_code == 200
        assert again.json()["created"] is False
        assert await session.scalar(select(func.count(Chunk.id))) == 1

    async def test_blank_document_is_rejected(self, client):
        response = await client.post("/documents", json={"title": "Empty", "text": "   "})

        assert response.status_code == 422
        assert response.json()["code"] == "invalid_document"

    async def test_markdown_upload(self, client):
        files = {"file": ("returns-policy.md", b"# Returns\n\nWithin 45 days.", "text/markdown")}

        response = await client.post("/documents/upload", files=files)

        assert response.status_code == 201
        assert response.json()["document"]["title"] == "Returns"  # from the # heading
        assert response.json()["document"]["source"] == "returns-policy.md"

    async def test_plain_text_upload_is_titled_after_the_file(self, client):
        files = {"file": ("office-hours.txt", b"We open at 9.", "text/plain")}

        response = await client.post("/documents/upload", files=files, data={"title": ""})

        assert response.json()["document"]["title"] == "office-hours"

    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("slides.pdf", b"%PDF-1.7"),
            ("notes.txt", "Café".encode("latin-1")),
            ("big.txt", b"a" * (2 * 1024 * 1024 + 1)),
        ],
    )
    async def test_bad_uploads_are_rejected(self, client, name, content):
        response = await client.post("/documents/upload", files={"file": (name, content)})

        assert response.status_code == 422
        assert response.json()["code"] == "invalid_document"


class TestEmbeddingsOnIngest:
    @pytest.fixture
    def embedder(self):
        return FakeEmbedder()

    async def test_chunks_are_embedded_when_a_provider_is_configured(
        self, client, session, embedder
    ):
        await client.post("/documents", json={"title": "Policy", "text": "Returns within 45 days."})

        chunk = await session.scalar(select(Chunk))
        assert chunk.embedding is not None
        assert embedder.calls == [("document", 1)]

    async def test_health_reports_hybrid_search(self, client):
        assert (await client.get("/health")).json() == {"status": "ok", "search": "hybrid"}


class TestAsk:
    async def test_answer_with_sources(self, client, claude, samples):
        claude.reply = claude_reply(
            {"type": "text", "text": "Refunds arrive "},
            cited(
                "within 5 business days after inspection",
                document_index=0,
                cited_text="Refunds go back to the original payment method within 5 business days",
            ),
            {"type": "text", "text": "."},
        )

        response = await client.post("/ask", json={"question": "When will my refund arrive?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Refunds arrive within 5 business days after inspection [1]."
        assert body["found"] is True
        assert body["sources"][0]["document_title"] == "Returns and Refunds"
        assert body["citations"][0]["number"] == 1
        assert body["usage"] == {"input_tokens": 1200, "output_tokens": 80}
        # The documents sent to Claude are the retrieved chunks, best first.
        documents = claude.requests[0]["messages"][0]["content"][:-1]
        assert documents[0]["title"].startswith("Returns and Refunds")

    async def test_unanswerable_question_skips_the_model(self, client, claude, samples):
        response = await client.post("/ask", json={"question": "quantum chromodynamics"})

        assert response.json()["found"] is False
        assert claude.requests == []

    async def test_search_endpoint_shows_retrieval_only(self, client, claude, samples):
        response = await client.post("/search", json={"query": "warranty claim photos", "top_k": 2})

        results = response.json()
        assert len(results) == 2
        assert results[0]["document_title"] == "Warranty"
        assert claude.requests == []

    async def test_model_outage_returns_502(self, client, claude, samples):
        claude.error = TypeError("Could not resolve authentication method.")

        response = await client.post("/ask", json={"question": "When will my refund arrive?"})

        assert response.status_code == 502
        assert response.json()["code"] == "llm_unavailable"


class TestApiKey:
    @pytest.fixture
    def settings(self, settings):
        return settings.model_copy(update={"api_key": SecretStr("s3cret")})

    async def test_key_is_required_when_configured(self, client):
        assert (await client.get("/documents")).status_code == 401
        assert (await client.get("/documents", headers={"X-API-Key": "nope"})).status_code == 401
        assert (await client.get("/documents", headers={"X-API-Key": "s3cret"})).status_code == 200

    async def test_health_stays_open(self, client):
        assert (await client.get("/health")).json() == {"status": "ok", "search": "keyword"}


class TestVoyageEmbedder:
    async def test_request_shape_batching_and_order(self, monkeypatch):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = __import__("json").loads(request.content)
            requests.append((request.headers["Authorization"], body))
            # Return items out of order to prove they are sorted by index.
            data = [
                {"object": "embedding", "embedding": [float(len(text))], "index": i}
                for i, text in enumerate(body["input"])
            ][::-1]
            return httpx.Response(200, json={"object": "list", "data": data})

        monkeypatch.setattr(VoyageEmbedder, "BATCH_SIZE", 2)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = VoyageEmbedder("voyage-key", "voyage-4", client=client)

        vectors = await embedder.embed(["a", "bb", "ccc"], input_type="document")

        assert vectors == [[1.0], [2.0], [3.0]]
        assert len(requests) == 2
        auth, body = requests[0]
        assert auth == "Bearer voyage-key"
        assert body == {
            "input": ["a", "bb"],
            "model": "voyage-4",
            "input_type": "document",
            "output_dimension": 1024,
        }

    async def test_provider_errors_become_embedding_errors(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        )
        embedder = VoyageEmbedder("voyage-key", "voyage-4", client=client)

        with pytest.raises(EmbeddingError):
            await embedder.embed(["text"], input_type="query")
