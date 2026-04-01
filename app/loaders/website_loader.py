"""
Website loader for roger.ink - manual trigger only.
Parses sitemap.xml and loads all pages.
"""

import xml.etree.ElementTree as ET

import httpx
from llama_index.core import Document

# Base website URL
WEBSITE_BASE_URL = "https://roger.ink"
SITEMAP_URL = "https://roger.ink/sitemap.xml"


async def fetch_sitemap() -> list[str]:
    """Fetch and parse sitemap.xml to get all page URLs."""
    async with httpx.AsyncClient() as client:
        response = await client.get(SITEMAP_URL, timeout=30.0)
        response.raise_for_status()

    # Parse XML
    root = ET.fromstring(response.text)

    # Extract URLs from sitemap
    urls = []
    for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
        loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc is not None and loc.text:
            urls.append(loc.text)

    return urls


async def fetch_page_content(url: str) -> str | None:
    """
    Fetch and extract text content from a page.
    Uses requests-html for JavaScript-rendered content if needed.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            # Simple HTML to text extraction
            html = response.text

            # Basic cleanup - remove script and style tags content
            import re

            # Remove script and style tags with content
            html = re.sub(
                r"<script[^>]*>.*?</script>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
            html = re.sub(
                r"<style[^>]*>.*?</style>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text

    except Exception:
        return None


async def load_website_documents() -> list[Document]:
    """
    Load all pages from roger.ink sitemap as LlamaIndex documents.

    Returns:
        List of Document objects with page content and metadata
    """
    # Fetch sitemap URLs
    urls = await fetch_sitemap()

    if not urls:
        return []

    documents = []

    for url in urls:
        # Fetch page content
        content = await fetch_page_content(url)

        if not content:
            continue

        # Create document
        document = Document(
            text=content,
            metadata={
                "source": "website_rogerink",
                "url": url,
                "title": _extract_title_from_url(url),
            },
        )

        documents.append(document)

    return documents


def _extract_title_from_url(url: str) -> str:
    """Extract a readable title from URL path."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        return "Home"

    # Convert path to title (e.g., "about/me" -> "About Me")
    parts = path.split("/")
    title = " ".join(parts).replace("-", " ").replace("_", " ")
    return title.title()
