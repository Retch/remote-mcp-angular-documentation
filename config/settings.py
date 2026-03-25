from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "angular_documentation"

    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_base_url: str = "http://localhost:11437"

    source_file: str = "./llms-full.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    min_score: float = 0.3
    docs_limit: int = 5

    fastmcp_port: int = 8000

    model_config = {"env_prefix": ""}


settings = Settings()
