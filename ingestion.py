from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FilterSelector, VectorParams, PointStruct
import structlog

qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "angular_documentation"
log = structlog.get_logger()
loader = TextLoader("./llms-full.txt")
docs = loader.load()
embeddings_model = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11437"
)
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
)
existing_collections = qdrant.get_collections().collections
collection_names = [c.name for c in existing_collections]

if collection_name not in collection_names:
    log.info(f"Collection {collection_name} not found - creating...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
else:
    log.info(f"Using existing collection {collection_name}")
    log.info(f"Clearing all existing points from collection {collection_name}")
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(must=[])
        )
    )

chunks = markdown_splitter.split_documents(docs)

log.info(f"Chunks: {len(chunks)}")

texts = [chunk.page_content for chunk in chunks[:1]]
embeddings = embeddings_model.embed_documents(texts)


points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"text": texts[i]}
    )
    for i in range(len(embeddings))
]

qdrant.upsert(
    collection_name=collection_name,
    points=points
)
