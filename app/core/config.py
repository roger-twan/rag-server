from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "postgresql://rag:rag@localhost:5432/rag_server_db"
    GITHUB_TOKEN: str = ""
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


settings = Settings()
