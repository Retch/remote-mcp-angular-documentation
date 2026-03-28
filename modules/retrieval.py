from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from config import settings
from type_definitions.retrieved_doc import RetrievedDoc

client = QdrantClient(url=settings.qdrant_url)

search_count = 0

embeddings_model = OllamaEmbeddings(
    model=settings.embedding_model, base_url=settings.embedding_base_url
)


def find_docs(
    query: str,
    angular_version: int,
    min_score: float = settings.min_score,
    docs_limit: int = settings.docs_limit,
    min_doc_length: int = settings.min_doc_length,
) -> list[RetrievedDoc]:
    global search_count
    search_count += 1
    query_vector = embeddings_model.embed_query(query)
    collection_name = settings.get_collection_name(angular_version)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=docs_limit,
        )
    except Exception as e:
        return []

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

def get_all_metrics() -> dict:
    global search_count
    total_docs = 0
    available_versions = []

    for version in settings.version_sources.keys():
        collection_name = settings.get_collection_name(version)
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            total_docs += collection_info.points_count
            available_versions.append(
                {
                    "version": version,
                    "document_count": collection_info.points_count,
                }
            )
        except Exception:
            pass

    return {
        "search_count": search_count,
        "total_documents": total_docs,
        "available_versions": available_versions,
    }
