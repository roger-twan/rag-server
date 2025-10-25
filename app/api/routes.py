from fastapi import APIRouter

from app.services.answer_generator import generate_answer

router = APIRouter()


@router.get("/query")
async def query(q: str):
    answer = await generate_answer(q)
    return {"query": q, "result": answer}
