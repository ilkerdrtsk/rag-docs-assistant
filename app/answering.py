"""Grounded answers from Claude, with citations back to the retrieved chunks.

Each retrieved chunk is sent as a `document` block with citations enabled. Claude's
reply comes back split into text blocks, and the cited ones point at the exact
document (chunk) and characters they rely on. That is turned into numbered
markers like "[1]" plus a list of sources the client can show.
"""

import logging
from dataclasses import dataclass, field

import anthropic

from app.errors import LLMRateLimited, LLMUnavailable
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You answer questions for a company's internal knowledge base. The user's message \
contains excerpts from the knowledge base as documents, followed by a question.

Answer only from those documents. If they do not contain the answer, say that the \
knowledge base does not cover it and stop; do not fill gaps with general knowledge. \
If the documents only partly answer the question, answer that part and say what is \
missing. Keep the answer short and direct, in the language of the question."""

NOT_FOUND = "I couldn't find anything about that in the knowledge base."
REFUSED = "I can't help with that request."

# Server-side fallback: if a safety classifier declines, the API retries the same request
# on a fallback model chosen by the refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: int
    document_title: str
    cited_text: str


@dataclass
class Answer:
    text: str
    found: bool
    citations: list[Citation] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def build_request(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """The user turn: one citable document per chunk, then the question."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": f"{chunk.document_title} (part {chunk.position + 1})",
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]
    return [{"role": "user", "content": [*documents, {"type": "text", "text": question}]}]


class AnswerGenerator:
    def __init__(
        self, client: anthropic.AsyncAnthropic, *, model: str, effort: str, max_tokens: int
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    async def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        if not chunks:
            # Nothing relevant was retrieved: answering anyway would invite a made-up reply.
            return Answer(text=NOT_FOUND, found=False)

        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=build_request(question, chunks),
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise LLMRateLimited("The language model is busy; try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            logger.warning("Claude API error %s: %s", exc.status_code, exc.message)
            raise LLMUnavailable("The language model could not answer right now.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable("The language model could not be reached.") from exc
        except TypeError as exc:
            # The SDK raises this before any network call when no credentials are configured.
            if "authentication method" not in str(exc):
                raise
            raise LLMUnavailable("Claude API credentials are not configured.") from exc

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            return Answer(text=REFUSED, found=False, model=response.model, **usage)

        parts: list[str] = []
        citations: list[Citation] = []
        numbers: dict[int, int] = {}  # chunk id -> marker number, in order of first use
        for block in response.content:
            if block.type != "text":
                continue  # thinking, fallback and other non-answer blocks
            parts.append(block.text)
            markers = []
            for cite in block.citations or []:
                if getattr(cite, "document_index", None) is None:
                    continue
                chunk = chunks[cite.document_index]
                number = numbers.setdefault(chunk.chunk_id, len(numbers) + 1)
                if number not in markers:
                    markers.append(number)
                citations.append(
                    Citation(number, chunk.chunk_id, chunk.document_title, cite.cited_text)
                )
            parts.extend(f" [{number}]" for number in markers)

        cited = [chunk for chunk in chunks if chunk.chunk_id in numbers]
        cited.sort(key=lambda chunk: numbers[chunk.chunk_id])
        return Answer(
            text="".join(parts).strip(),
            found=bool(citations),
            citations=citations,
            sources=cited,
            model=response.model,
            **usage,
        )
