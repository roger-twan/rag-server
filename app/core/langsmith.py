import os

from app.core.config import settings


def configure_langsmith_tracing() -> None:
    values = {
        "LANGSMITH_TRACING": "true" if settings.LANGSMITH_TRACING else "false",
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": settings.LANGSMITH_PROJECT,
        "LANGSMITH_ENDPOINT": settings.LANGSMITH_ENDPOINT,
        "LANGSMITH_WORKSPACE_ID": settings.LANGSMITH_WORKSPACE_ID,
        "LANGCHAIN_CALLBACKS_BACKGROUND": settings.LANGCHAIN_CALLBACKS_BACKGROUND,
    }

    for key, value in values.items():
        if value:
            os.environ[key] = value
