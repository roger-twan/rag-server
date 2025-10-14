from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    QDRANT_URL: str
    QDRANT_API_KEY: str
    GITHUB_TOKEN: str
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
