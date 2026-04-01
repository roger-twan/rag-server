"""Markdown-aware chunking strategies for blog posts."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MarkdownChunk:
    """Represents a semantic chunk of a markdown document."""

    text: str
    level: int  # Header level (0 for no header, 1 for #, 2 for ##, etc.)
    header: str  # The header text
    parent_headers: list[str]  # Parent headers for context
    char_start: int  # Start position in original text
    char_end: int  # End position in original text


def chunk_by_headers(
    text: str,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
) -> list[MarkdownChunk]:
    """
    Chunk markdown text by headers, respecting document structure.

    Strategy:
    1. Split by headers (#, ##, ###, ####)
    2. Merge small sections with siblings
    3. Split large sections at paragraph boundaries

    Args:
        text: Markdown content (frontmatter already removed)
        min_chunk_size: Minimum chunk size to avoid tiny chunks
        max_chunk_size: Maximum chunk size before forced splitting

    Returns:
        List of MarkdownChunk objects with metadata
    """
    # Parse headers and their levels
    header_pattern = r"^(#{1,4})\s+(.+)$"

    lines = text.split("\n")
    sections = []
    current_section = []
    current_header = ""
    current_level = 0
    current_parent_headers = []
    char_pos = 0
    section_start = 0

    # Track header hierarchy
    header_stack: list[tuple[int, str]] = []  # (level, header)

    for i, line in enumerate(lines):
        header_match = re.match(header_pattern, line)

        if header_match:
            # Save previous section if exists
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append(
                        MarkdownChunk(
                            text=section_text,
                            level=current_level,
                            header=current_header,
                            parent_headers=current_parent_headers.copy(),
                            char_start=section_start,
                            char_end=char_pos,
                        )
                    )

            # Start new section with this header
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()

            # Update header stack - pop headers with >= level
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            # Current parent headers are what's left in stack
            current_parent_headers = [h[1] for h in header_stack]

            # Push current header
            header_stack.append((level, header_text))

            current_level = level
            current_header = header_text
            current_section = [line]
            section_start = char_pos
        else:
            current_section.append(line)

        char_pos += len(line) + 1  # +1 for newline

    # Don't forget the last section
    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append(
                MarkdownChunk(
                    text=section_text,
                    level=current_level,
                    header=current_header,
                    parent_headers=current_parent_headers.copy(),
                    char_start=section_start,
                    char_end=char_pos,
                )
            )

    # Post-processing: merge small sections and split large ones
    return _optimize_chunks(sections, min_chunk_size, max_chunk_size)


def _optimize_chunks(
    chunks: list[MarkdownChunk],
    min_size: int,
    max_size: int,
) -> list[MarkdownChunk]:
    """Merge small chunks and split large ones."""
    if not chunks:
        return []

    optimized: list[MarkdownChunk] = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]
        chunk_text = chunk.text

        # Handle oversized chunks
        if len(chunk_text) > max_size:
            # Split at paragraph boundaries
            sub_chunks = _split_large_chunk(chunk, max_size)
            optimized.extend(sub_chunks)
            i += 1
            continue

        # Try to merge with next chunk if too small
        current_merge = chunk_text
        current_headers = [chunk.header] if chunk.header else []
        merge_level = chunk.level
        j = i + 1

        while j < len(chunks) and len(current_merge) < min_size:
            next_chunk = chunks[j]
            # Only merge if same header level or next is subsection
            if next_chunk.level >= merge_level:
                current_merge += "\n\n" + next_chunk.text
                if next_chunk.header:
                    current_headers.append(next_chunk.header)
                j += 1
            else:
                break

        if j > i + 1:  # We merged some chunks
            merged_chunk = MarkdownChunk(
                text=current_merge,
                level=merge_level,
                header=chunk.header,
                parent_headers=chunk.parent_headers,
                char_start=chunk.char_start,
                char_end=chunks[j - 1].char_end,
            )
            optimized.append(merged_chunk)
            i = j
        else:
            optimized.append(chunk)
            i += 1

    return optimized


def _split_large_chunk(chunk: MarkdownChunk, max_size: int) -> list[MarkdownChunk]:
    """Split a large chunk at paragraph boundaries."""
    paragraphs = chunk.text.split("\n\n")
    result = []
    current_text = ""
    current_start = chunk.char_start

    for para in paragraphs:
        if len(current_text) + len(para) > max_size and current_text:
            # Save current chunk
            result.append(
                MarkdownChunk(
                    text=current_text.strip(),
                    level=chunk.level,
                    header=chunk.header,
                    parent_headers=chunk.parent_headers,
                    char_start=current_start,
                    char_end=current_start + len(current_text),
                )
            )
            current_start += len(current_text) + 2  # +2 for \n\n
            current_text = para
        else:
            if current_text:
                current_text += "\n\n"
            current_text += para

    # Don't forget the last piece
    if current_text:
        result.append(
            MarkdownChunk(
                text=current_text.strip(),
                level=chunk.level,
                header=chunk.header,
                parent_headers=chunk.parent_headers,
                char_start=current_start,
                char_end=chunk.char_end,
            )
        )

    return result


def enrich_chunk_context(
    chunk: MarkdownChunk,
    document_title: str = "",
    include_parents: bool = True,
) -> str:
    """
    Enrich chunk text with context for better embedding.

    Adds document title and parent headers to chunk text.

    Args:
        chunk: The markdown chunk
        document_title: The overall document/blog title
        include_parents: Whether to include parent headers

    Returns:
        Enriched text suitable for embedding
    """
    context_parts = []

    if document_title:
        context_parts.append(f"Document: {document_title}")

    if include_parents and chunk.parent_headers:
        context_parts.append(f"Section: {' > '.join(chunk.parent_headers)}")

    if chunk.header:
        context_parts.append(f"Subsection: {chunk.header}")

    if context_parts:
        return "\n".join(context_parts) + "\n\n" + chunk.text

    return chunk.text


def chunk_markdown_document(
    text: str,
    document_title: str = "",
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
    enrich_context: bool = True,
) -> list[dict[str, Any]]:
    """
    High-level function to chunk a markdown document with full metadata.

    Args:
        text: Markdown content (frontmatter removed)
        document_title: Blog post title for context
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        enrich_context: Whether to add title/parent headers to chunk text

    Returns:
        List of dicts with 'text', 'metadata', 'char_start', 'char_end'
    """
    chunks = chunk_by_headers(text, min_chunk_size, max_chunk_size)

    result = []
    for chunk in chunks:
        # Build chunk text (with or without context enrichment)
        if enrich_context:
            final_text = enrich_chunk_context(chunk, document_title, include_parents=True)
        else:
            final_text = chunk.text

        result.append(
            {
                "text": final_text,
                "metadata": {
                    "header": chunk.header,
                    "header_level": chunk.level,
                    "parent_headers": " > ".join(chunk.parent_headers),  # Convert list to string
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "original_length": len(chunk.text),
                },
            }
        )

    return result
