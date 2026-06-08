from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API keys
    JWT_SECRET: str = "change-me-in-prod"
    GROQ_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "true"

    # Service URLs
    GPTCACHE_URL: str = "http://gptcache:8001"
    POSTGRES_DSN: str = "postgresql://postgres:postgres@localhost:5432/policy_bot"

    # Retrieval: local FAISS index directory (built by ingestion pipeline)
    FAISS_INDEX_DIR: str = "faiss_index"

    # Tuning
    MAX_INPUT_CHARS: int = 4000
    MAX_SESSION_TURNS: int = 10
    FAITHFULNESS_THRESHOLD: float = 0.7
    COMPLETENESS_THRESHOLD: float = 0.6

    # Model — single Groq model used everywhere (low + high + judges + tree search)
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    class Config:
        env_file = ".env"


settings = Settings()
