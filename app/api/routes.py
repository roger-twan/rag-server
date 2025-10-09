from fastapi import APIRouter

from app.services.retriever import retrieve

router = APIRouter()


@router.get("/query")
async def query(q: str):
    chunks = await retrieve(q)
    return {"query": q, "result": chunks}
