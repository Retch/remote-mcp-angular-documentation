from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from type_definitions.retrieved_doc import RetrievedDoc

client = QdrantClient(url="http://localhost:6333")

embeddings_model = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11437"
)

def find_docs(query: str, min_score: float = 0.4, docs_limit: int = 5) -> list[RetrievedDoc]:
    query_vector = embeddings_model.embed_query(query)

    results = client.query_points(
        collection_name="angular_documentation",
        query=query_vector,
        limit=docs_limit
    )

    filtered_results = [
        p for p in results.points
        if p.score >= min_score
    ]

    filtered_docs: list[RetrievedDoc] = []

    for p in filtered_results:
        doc: RetrievedDoc = RetrievedDoc(text=p.payload.get("text"), score=p.score)
        filtered_docs.append(doc)
    return filtered_docs
