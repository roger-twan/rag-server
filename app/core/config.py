from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    GITHUB_TOKEN: str
    GITHUB_WEBHOOK_SECRET: str
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str
    COHERE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_HOST: str

    class Config:
        env_file = ".env"


settings = Settings()
