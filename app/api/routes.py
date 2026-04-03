from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.scripts.sync_jobs import sync_all_github_repos, sync_notes, sync_website
from app.services.answer_generator import generate_answer

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def verify_public_token(x_api_token: str = Header(...)):
    if x_api_token != settings.PUBLIC_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid public API token",
        )
    return x_api_token


def verify_admin_token(x_api_token: str = Header(...)):
    if x_api_token != settings.ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API token",
        )
    return x_api_token


@router.get("/query")
@limiter.limit("10/minute")
async def query(request: Request, q: str, token: str = Depends(verify_public_token)):
    answer = await generate_answer(q)
    return {"query": q, "result": answer}


@router.post("/ingest/website")
@limiter.limit("5/hour")
async def ingest_website(request: Request, token: str = Depends(verify_admin_token)):
    """
    Manually trigger website ingestion (roger.ink).
    Fetches sitemap, loads all pages, and indexes to Pinecone.
    """
    result = await sync_website()
    return result


@router.post("/ingest/github-all-repos")
@limiter.limit("5/hour")
async def ingest_github_repos(request: Request, token: str = Depends(verify_admin_token)):
    """
    Batch ingest all GitHub repos (except notes) to Pinecone.
    For each repo, extracts: description, README, package.json
    """
    result = await sync_all_github_repos()
    return result


@router.post("/ingest/notes")
@limiter.limit("5/hour")
async def ingest_notes(request: Request, token: str = Depends(verify_admin_token)):
    """
    Manually trigger notes repo (blog) ingestion.
    Loads Portfolio, Technical directories and Skills.md, then re-indexes to Pinecone.
    """
    result = await sync_notes()
    return result
