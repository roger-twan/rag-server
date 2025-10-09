from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI()

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "RAG Server is running", "env": settings.ENVIRONMENT}
