import uuid

import anthropic
import httpx2
import pytest

from app.answering import FALLBACK_BETA, NOT_FOUND, REFUSED, SYSTEM_PROMPT
from app.errors import LLMRateLimited, LLMUnavailable
from app.retrieval import RetrievedChunk
from tests.conftest import cited, claude_reply


def chunk(chunk_id: int, title: str, text: str, position: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        document_title=title,
        position=position,
        text=text,
        score=0.03,
        keyword_rank=1,
        vector_rank=None,
    )


CHUNKS = [
    chunk(10, "Returns and Refunds", "Refunds take 5 business days.", position=2),
    chunk(20, "Shipping", "Express shipping costs 12 EUR."),
]


async def test_request_sends_each_chunk_as_a_citable_document(generator, claude):
    await generator.answer("How long do refunds take?", CHUNKS)

    [request] = claude.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == SYSTEM_PROMPT
    assert request["output_config"] == {"effort": "medium"}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert not {"temperature", "top_p", "top_k"} & request.keys()

    [message] = request["messages"]
    assert message["role"] == "user"
    *documents, question = message["content"]
    assert question == {"type": "text", "text": "How long do refunds take?"}
    assert documents[0] == {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": "Refunds take 5 business days.",
        },
        "title": "Returns and Refunds (part 3)",
        "citations": {"enabled": True},
    }
    assert len(documents) == 2


async def test_citations_become_numbered_markers(generator, claude):
    claude.reply = claude_reply(
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Refunds take "},
        cited("5 business days", document_index=0, cited_text="Refunds take 5 business days."),
        {"type": "text", "text": ", and express shipping costs "},
        cited("12 EUR", document_index=1, cited_text="Express shipping costs 12 EUR."),
        {"type": "text", "text": ". Again: "},
        cited("5 days", document_index=0, cited_text="Refunds take 5 business days."),
        {"type": "text", "text": "."},
    )

    answer = await generator.answer("question", CHUNKS)

    assert answer.text == (
        "Refunds take 5 business days [1], and express shipping costs 12 EUR [2]. "
        "Again: 5 days [1]."
    )
    assert answer.found is True
    assert [c.number for c in answer.citations] == [1, 2, 1]
    assert [source.chunk_id for source in answer.sources] == [10, 20]
    assert (answer.input_tokens, answer.output_tokens) == (1200, 80)


async def test_answer_without_citations_is_not_found(generator, claude):
    claude.reply = claude_reply({"type": "text", "text": "The knowledge base does not cover it."})

    answer = await generator.answer("question", CHUNKS)

    assert answer.found is False
    assert answer.sources == []


async def test_nothing_retrieved_means_no_model_call(generator, claude):
    answer = await generator.answer("question", [])

    assert answer.text == NOT_FOUND
    assert answer.found is False
    assert claude.requests == []


async def test_refusal_is_reported_without_reading_content(generator, claude):
    claude.reply = claude_reply(stop_reason="refusal")

    answer = await generator.answer("question", CHUNKS)

    assert (answer.text, answer.found) == (REFUSED, False)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            anthropic.RateLimitError(
                "busy", response=httpx2.Response(429, request=REQUEST), body=None
            ),
            LLMRateLimited,
        ),
        (
            anthropic.InternalServerError(
                "down", response=httpx2.Response(500, request=REQUEST), body=None
            ),
            LLMUnavailable,
        ),
        (anthropic.APIConnectionError(request=REQUEST), LLMUnavailable),
        (TypeError("Could not resolve authentication method."), LLMUnavailable),
    ],
)
async def test_api_failures_become_clear_errors(generator, claude, error, expected):
    claude.error = error

    with pytest.raises(expected):
        await generator.answer("question", CHUNKS)


async def test_unrelated_type_errors_are_not_hidden(generator, claude):
    claude.error = TypeError("a real bug")

    with pytest.raises(TypeError, match="a real bug"):
        await generator.answer("question", CHUNKS)
