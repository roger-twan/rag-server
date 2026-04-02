"""Markdown frontmatter parser for extracting YAML metadata."""

import re
from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Frontmatter format:
    ---
    title: "Post Title"
    date: "2024-01-01"
    tags: ["tag1", "tag2"]
    ---

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    # Pattern to match frontmatter: ---\n...\n---
    pattern = r"^---\s*\n(.*?)\n---\s*\n?(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body


def extract_blog_metadata(content: str, file_path: str = "") -> dict[str, Any]:
    """
    Extract blog-specific metadata from markdown content.

    Args:
        content: Full markdown content including frontmatter
        file_path: Optional file path for fallback title

    Returns:
        Dict with title, date, tags, and other metadata
    """
    frontmatter, body = parse_frontmatter(content)

    metadata = {
        "title": frontmatter.get("title", ""),
        "date": frontmatter.get("date", ""),
        "tags": frontmatter.get("tags", []),
        "description": frontmatter.get("description", ""),
        "author": frontmatter.get("author", ""),
        "category": frontmatter.get("category", ""),
        "publish": frontmatter.get("publish", False),
    }

    # Fallback: extract title from first H1 if frontmatter has no title
    if not metadata["title"]:
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if h1_match:
            metadata["title"] = h1_match.group(1).strip()
        elif file_path:
            # Use filename as fallback
            metadata["title"] = (
                file_path.split("/")[-1].replace(".md", "").replace("-", " ").replace("_", " ")
            )

    return metadata
