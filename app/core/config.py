from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "postgresql://rag:rag@localhost:5432/rag_server_db"
    GITHUB_TOKEN: str = ""
    GITHUB_HTTP_TIMEOUT_SECONDS: float = 30.0
    GITHUB_HTTP_RETRIES: int = 3
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_HOST: str = ""
    ENABLE_SPARSE_SEARCH: bool = False
    LLM_PROVIDER: str = "google"  # "google", "openai", or "deepseek"
    DEEPSEEK_API_KEY: Optional[str] = None
    GITHUB_WEBHOOK_SECRET: Optional[str] = None
    PUBLIC_API_TOKEN: str = ""
    ADMIN_API_TOKEN: str = ""
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "rag-server"
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_WORKSPACE_ID: str = ""
    LANGCHAIN_CALLBACKS_BACKGROUND: str = ""
    RAGAS_LLM_MODEL: str = "gpt-4o-mini"
    RAGAS_EMBEDDING_MODEL: str = "text-embedding-3-small"


settings = Settings()
