import pytest

from app.chunking import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("Just one paragraph.")

    assert [(chunk.position, chunk.text) for chunk in chunks] == [(0, "Just one paragraph.")]


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n\n  ") == []


def test_paragraphs_are_packed_without_exceeding_the_limit():
    paragraphs = [f"Paragraph {n} " + "x" * 80 for n in range(20)]

    chunks = chunk_text("\n\n".join(paragraphs), max_chars=300, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    assert all(paragraph in joined for paragraph in paragraphs)


def test_each_chunk_starts_with_its_section_heading():
    text = "# Refunds\n\n" + "\n\n".join(f"Rule {n}: " + "y" * 90 for n in range(6))

    chunks = chunk_text(text, max_chars=250, overlap=0)

    assert len(chunks) > 1
    assert all(chunk.text.startswith("# Refunds\n\n") for chunk in chunks)


def test_a_new_heading_starts_a_new_chunk():
    chunks = chunk_text("# A\n\nFirst section.\n\n# B\n\nSecond section.")

    assert [chunk.text for chunk in chunks] == ["# A\n\nFirst section.", "# B\n\nSecond section."]


def test_heading_without_body_is_not_a_chunk():
    chunks = chunk_text("# Title\n\n## Section\n\nBody text.")

    assert [chunk.text for chunk in chunks] == ["## Section\n\nBody text."]


def test_consecutive_chunks_overlap():
    paragraphs = [f"Sentence number {n} about returns." for n in range(12)]

    chunks = chunk_text("\n\n".join(paragraphs), max_chars=160, overlap=60)

    for previous, following in zip(chunks, chunks[1:], strict=False):
        last_paragraph = previous.text.split("\n\n")[-1]
        assert following.text.startswith(last_paragraph)


def test_long_paragraphs_are_split_by_sentence_then_by_width():
    long_paragraph = "Short sentence. " + "z" * 500 + ". Final sentence."

    chunks = chunk_text(long_paragraph, max_chars=200, overlap=0)

    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert chunks[0].text.startswith("Short sentence.")
    assert chunks[-1].text.endswith("Final sentence.")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=100, overlap=100)
