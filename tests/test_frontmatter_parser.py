"""Tests for frontmatter parser."""

from app.utils.frontmatter_parser import (
    extract_blog_metadata,
    parse_frontmatter,
)


class TestParseFrontmatter:
    """Tests for parse_frontmatter function."""

    def test_parses_valid_frontmatter(self):
        """Test parsing markdown with valid YAML frontmatter."""
        content = """---
title: "My Blog Post"
date: "2024-01-15"
tags: ["python", "fastapi"]
---
# Main Title

This is the content."""

        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["title"] == "My Blog Post"
        assert frontmatter["date"] == "2024-01-15"
        assert frontmatter["tags"] == ["python", "fastapi"]
        assert "# Main Title" in body
        assert "This is the content." in body
        assert "---" not in body

    def test_no_frontmatter_returns_empty_dict(self):
        """Test that content without frontmatter returns empty dict."""
        content = "# Just a title\n\nSome content."

        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_empty_frontmatter(self):
        """Test handling of empty frontmatter."""
        content = "---\n---\n# Content"

        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert "# Content" in body

    def test_multiline_description(self):
        """Test parsing frontmatter with multiline description."""
        content = """---
title: "Post"
description: |
  This is a long description
  that spans multiple lines.
---
# Content"""

        frontmatter, body = parse_frontmatter(content)

        assert "long description" in frontmatter["description"]
        assert "multiple lines" in frontmatter["description"]


class TestExtractBlogMetadata:
    """Tests for extract_blog_metadata function."""

    def test_extracts_all_metadata(self):
        """Test extracting complete metadata from frontmatter."""
        content = """---
title: "My Post"
date: "2024-01-15"
tags: ["python", "ai"]
description: "A great post"
author: "Roger"
category: "Technical"
---
# Content"""

        metadata = extract_blog_metadata(content)

        assert metadata["title"] == "My Post"
        assert metadata["date"] == "2024-01-15"
        assert metadata["tags"] == ["python", "ai"]
        assert metadata["description"] == "A great post"
        assert metadata["author"] == "Roger"
        assert metadata["category"] == "Technical"

    def test_fallback_to_h1_title(self):
        """Test falling back to H1 when no frontmatter title."""
        content = """---
date: "2024-01-15"
---
# My Awesome Post

Content here."""

        metadata = extract_blog_metadata(content)

        assert metadata["title"] == "My Awesome Post"
        assert metadata["date"] == "2024-01-15"

    def test_fallback_to_filename(self):
        """Test falling back to filename when no title in content."""
        content = "Just some content without headers."

        metadata = extract_blog_metadata(content, file_path="posts/my-blog-post.md")

        assert metadata["title"] == "my blog post"

    def test_empty_tags_default(self):
        """Test that missing tags default to empty list."""
        content = """---
title: "Post"
---
Content"""

        metadata = extract_blog_metadata(content)

        assert metadata["tags"] == []

    def test_handles_invalid_yaml(self):
        """Test graceful handling of invalid YAML."""
        content = """---
not valid yaml: [unclosed
---
# Title"""

        # Should not raise exception
        metadata = extract_blog_metadata(content)

        # Title should be extracted from H1
        assert metadata["title"] == "Title"
