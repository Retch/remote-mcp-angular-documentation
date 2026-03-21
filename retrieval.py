from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

client = QdrantClient(url="http://localhost:6333")

embeddings_model = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11437"
)

query = "ngClass"

query_vector = embeddings_model.embed_query(query)

results = client.query_points(
    collection_name="angular_documentation",
    query=query_vector,
    limit=5
)

threshold = 0.4

filtered_results = [
    p for p in results.points
    if p.score >= threshold
]

for p in filtered_results:
    print("Score:", p.score)
    print("Text:", p.payload.get("text"))
    print()
