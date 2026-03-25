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
) -> list[RetrievedDoc]:
    query_vector = embeddings_model.embed_query(query)

    results = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=docs_limit,
    )

    filtered_results = [p for p in results.points if p.score >= min_score]

    filtered_docs: list[RetrievedDoc] = []

    for p in filtered_results:
        text = p.payload["text"] if p.payload else ""
        doc: RetrievedDoc = RetrievedDoc(text=text, score=p.score)
        filtered_docs.append(doc)
    return filtered_docs
