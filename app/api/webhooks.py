"""
GitHub webhook handler for notes repo real-time sync.
"""

import logging

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.loaders.github_loader import NotesRepoLoader, is_notes_repo_push, verify_github_webhook
from app.services.ingestion import ingest_documents_batch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
):
    """
    Handle GitHub webhook for notes repo pushes.
    Verifies signature, checks if it's notes repo main branch push, then re-indexes.
    """
    # Read raw body for signature verification
    body = await request.body()

    # Verify webhook signature
    if not verify_github_webhook(body, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Parse payload
    try:
        import json

        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # Check if this is notes repo push to main
    if not is_notes_repo_push(payload):
        logger.info("Webhook received but not a notes repo main branch push")
        return {
            "status": "ignored",
            "reason": "Not a notes repo main branch push",
        }

    # Load documents from notes repo
    try:
        documents = NotesRepoLoader.load_documents()

        if not documents:
            logger.warning("No documents found in notes repo")
            return {
                "status": "no_data",
                "message": "No documents found to index",
            }

        # Re-index (clear existing and add new)
        result = await ingest_documents_batch(
            documents=documents,
            source="github_notes",
            clear_existing=True,
        )

        logger.info(
            f"Successfully re-indexed notes repo: {result['total_chunks']} chunks from {result['documents_indexed']} documents"
        )

        return {
            "status": "success",
            "message": f"Re-indexed {result['documents_indexed']} documents with {result['total_chunks']} chunks",
            "details": result,
        }

    except Exception as e:
        logger.error(f"Failed to re-index notes repo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-index: {str(e)}",
        )
