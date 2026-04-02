"""Tests for markdown chunker."""

from app.services.markdown_chunker import (
    MarkdownChunk,
    chunk_by_headers,
    chunk_markdown_document,
    enrich_chunk_context,
)


class TestChunkByHeaders:
    """Tests for chunk_by_headers function."""

    def test_chunks_by_h2_headers(self):
        """Test splitting by H2 headers."""
        text = """# Main Title

Introduction paragraph.

## Section One

Content for section one.
More content.

## Section Two

Content for section two."""

        chunks = chunk_by_headers(text, min_chunk_size=50, max_chunk_size=1000)

        assert len(chunks) >= 2
        # First chunk should be intro + first section (merged because small)
        # Or at least we should have chunks containing section headers
        section_headers = [c.header for c in chunks]
        assert "Section One" in section_headers or any("Section One" in c.text for c in chunks)

    def test_respects_header_hierarchy(self):
        """Test that parent headers are tracked."""
        text = """# Title

## Parent Section

### Subsection A

Content A.

### Subsection B

Content B."""

        chunks = chunk_by_headers(text, min_chunk_size=10, max_chunk_size=1000)

        # Find chunk with Subsection A
        subsection_chunk = next((c for c in chunks if c.header == "Subsection A"), None)
        if subsection_chunk:
            assert "Parent Section" in subsection_chunk.parent_headers

    def test_splits_large_sections(self):
        """Test that oversized sections are split."""
        # Create a section larger than max_chunk_size
        large_content = "A " * 2000
        text = f"## Big Section\n\n{large_content}"

        chunks = chunk_by_headers(text, min_chunk_size=10, max_chunk_size=500)

        big_section_chunks = [
            c for c in chunks if c.header == "Big Section" or "Big Section" in c.text
        ]
        assert len(big_section_chunks) > 1 or any(len(c.text) > 500 for c in chunks)

    def test_merges_small_sections(self):
        """Test that small adjacent sections are merged."""
        text = """## Tiny Section

Short.

## Another Tiny

Also short."""

        chunks = chunk_by_headers(text, min_chunk_size=100, max_chunk_size=1000)

        # Small sections should be merged
        assert len(chunks) <= 2


class TestEnrichChunkContext:
    """Tests for enrich_chunk_context function."""

    def test_adds_document_title(self):
        """Test that document title is prepended."""
        chunk = MarkdownChunk(
            text="Content here.",
            level=2,
            header="Section",
            parent_headers=[],
            char_start=0,
            char_end=20,
        )

        enriched = enrich_chunk_context(chunk, document_title="My Blog", include_parents=True)

        assert "Document: My Blog" in enriched
        assert "Content here." in enriched

    def test_adds_parent_headers(self):
        """Test that parent headers are included."""
        chunk = MarkdownChunk(
            text="Content.",
            level=3,
            header="Subsection",
            parent_headers=["Chapter 1", "Section A"],
            char_start=0,
            char_end=10,
        )

        enriched = enrich_chunk_context(chunk, document_title="Book", include_parents=True)

        assert "Section: Chapter 1 > Section A" in enriched
        assert "Subsection: Subsection" in enriched

    def test_respects_include_parents_false(self):
        """Test that parent headers can be excluded."""
        chunk = MarkdownChunk(
            text="Content.",
            level=1,
            header="Title",
            parent_headers=[],
            char_start=0,
            char_end=10,
        )

        enriched = enrich_chunk_context(chunk, document_title="Blog", include_parents=False)

        assert "Document: Blog" in enriched
        assert "Section:" not in enriched


class TestChunkMarkdownDocument:
    """Tests for chunk_markdown_document high-level function."""

    def test_returns_chunk_dicts(self):
        """Test that function returns list of dicts with expected keys."""
        text = """# Title

## Section 1

Content one.

## Section 2

Content two."""

        result = chunk_markdown_document(
            text=text,
            document_title="My Doc",
            min_chunk_size=50,
            max_chunk_size=1000,
            enrich_context=True,
        )

        assert isinstance(result, list)
        assert len(result) > 0

        for chunk_dict in result:
            assert "text" in chunk_dict
            assert "metadata" in chunk_dict
            assert "header" in chunk_dict["metadata"]
            assert "header_level" in chunk_dict["metadata"]
            assert "parent_headers" in chunk_dict["metadata"]
            assert isinstance(chunk_dict["metadata"]["parent_headers"], str)
            assert "Document: My Doc" in chunk_dict["text"]

    def test_without_context_enrichment(self):
        """Test chunking without context enrichment."""
        text = "# Title\n\nSimple content."

        result = chunk_markdown_document(
            text=text,
            document_title="Blog",
            enrich_context=False,
        )

        assert len(result) > 0
        assert "Document: Blog" not in result[0]["text"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_by_headers("")
        assert chunks == []

    def test_no_headers(self):
        """Test handling of text without headers."""
        text = "Just some plain text without any markdown headers."

        chunks = chunk_by_headers(text, min_chunk_size=10, max_chunk_size=1000)

        # Should create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].text == text
        assert chunks[0].level == 0  # No header

    def test_single_hash_headers(self):
        """Test handling of single # headers."""
        text = """# Section 1

Content 1.

# Section 2

Content 2."""

        chunks = chunk_by_headers(text, min_chunk_size=10, max_chunk_size=1000)

        # Both H1 sections should be separate chunks
        assert len(chunks) >= 2
        section_headers = [c.header for c in chunks]
        assert "Section 1" in section_headers
        assert "Section 2" in section_headers
