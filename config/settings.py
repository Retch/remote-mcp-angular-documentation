from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    base_collection_name: str = "angular_documentation"

    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_base_url: str = "http://localhost:11437"

    version_sources: dict[int, str] = {
        21: "/app/llms-full/21.txt",
    }
    default_version: int = 21
    
    chunk_size: int = 1000
    chunk_overlap: int = 200

    min_score: float = 0.3
    docs_limit: int = 5
    min_doc_length: int = 5

    fastmcp_port: int = 8000

    model_config = {"env_prefix": ""}

    def get_collection_name(self, version: int) -> str:
        return f"{self.base_collection_name}_v{version}"

    def get_source_file(self, version: int) -> str | None:
        return self.version_sources.get(version)


settings = Settings()
