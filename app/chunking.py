"""Split documents into overlapping, heading-aware chunks for retrieval.

Paragraphs are packed into chunks of at most `max_chars`. Each chunk starts with
the Markdown heading it belongs to, so a chunk like "Items must be unused..." is
still found by a search for "returns policy". Consecutive chunks share up to
`overlap` characters so an answer split across a boundary is not lost.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    position: int
    text: str


def _pieces(text: str, max_chars: int) -> list[str]:
    """Paragraphs, with over-long paragraphs broken into sentences, then into fixed widths."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_chars:
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence.strip():
                pieces.append(sentence.strip())
    return pieces


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[TextChunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[str] = []
    current: list[str] = []
    heading: str | None = None

    def size(parts: list[str]) -> int:
        return sum(len(part) for part in parts) + 2 * max(len(parts) - 1, 0)

    def start_new_chunk() -> None:
        nonlocal current
        chunks.append("\n\n".join(current))
        # Carry the tail of the previous chunk forward as overlap.
        tail: list[str] = []
        for part in reversed(current):
            if part == heading or size([part, *tail]) > overlap:
                break
            tail.insert(0, part)
        current = ([heading] if heading else []) + tail

    for piece in _pieces(text, max_chars):
        if piece.startswith("#"):
            if current and current != [heading]:
                start_new_chunk()
            heading = piece
            current = [heading]
            continue
        if current and size([*current, piece]) > max_chars:
            start_new_chunk()
            # The overlap plus this piece may still be too big; then drop the overlap.
            if size([*current, piece]) > max_chars:
                current = [heading] if heading and size([heading, piece]) <= max_chars else []
        current.append(piece)

    if current and current != [heading]:
        chunks.append("\n\n".join(current))
    return [TextChunk(position=index, text=chunk) for index, chunk in enumerate(chunks)]
