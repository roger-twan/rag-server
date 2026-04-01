from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    ENVIRONMENT: str = "development"
    GITHUB_TOKEN: str
    GITHUB_WEBHOOK_SECRET: Optional[str] = None
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str
    COHERE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_HOST: str


settings = Settings()
