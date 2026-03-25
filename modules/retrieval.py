from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from config import settings
from type_definitions.retrieved_doc import RetrievedDoc

client = QdrantClient(url=settings.qdrant_url)

embeddings_model = OllamaEmbeddings(
    model=settings.embedding_model, base_url=settings.embedding_base_url
)

def find_docs(
    query: str,
    min_score: float = settings.min_score,
    docs_limit: int = settings.docs_limit,
    min_doc_length: int = settings.min_doc_length,
) -> list[RetrievedDoc]:
    query_vector = embeddings_model.embed_query(query)

    results = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=docs_limit,
    )

    filtered_results = [
        p
        for p in results.points
        if p.score >= min_score and len(p.payload.get("text", "")) >= min_doc_length
    ]

    filtered_docs: list[RetrievedDoc] = []

    for p in filtered_results:
        text = p.payload.get("text", "")
        doc: RetrievedDoc = RetrievedDoc(text=text, score=p.score)
        filtered_docs.append(doc)
    return filtered_docs
