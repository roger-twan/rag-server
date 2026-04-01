from fastapi import APIRouter

from app.scripts.sync_jobs import sync_all_github_repos, sync_notes, sync_website
from app.services.answer_generator import generate_answer

router = APIRouter()


@router.get("/query")
async def query(q: str):
    answer = await generate_answer(q)
    return {"query": q, "result": answer}


@router.post("/ingest/website")
async def ingest_website():
    """
    Manually trigger website ingestion (roger.ink).
    Fetches sitemap, loads all pages, and indexes to Pinecone.
    """
    result = await sync_website()
    return result


@router.post("/ingest/github-all-repos")
async def ingest_github_repos():
    """
    Batch ingest all GitHub repos (except notes) to Pinecone.
    For each repo, extracts: description, README, package.json
    """
    result = await sync_all_github_repos()
    return result


@router.post("/ingest/notes")
async def ingest_notes():
    """
    Manually trigger notes repo (blog) ingestion.
    Loads Portfolio, Technical directories and Skills.md, then re-indexes to Pinecone.
    """
    result = await sync_notes()
    return result
