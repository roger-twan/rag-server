from fastapi import FastAPI

from app.api.routes import router as api_router
from app.api.webhooks import router as webhook_router
from app.core.config import settings
from app.core.events import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api")
app.include_router(webhook_router, prefix="/api/webhooks")


@app.get("/")
def root():
    return {"message": "RAG Server is running", "env": settings.ENVIRONMENT}
