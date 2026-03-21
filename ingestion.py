from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FilterSelector,
    VectorParams,
    PointStruct,
)
import structlog
from tqdm import tqdm
from datetime import datetime, UTC
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

qdrant = QdrantClient(url=config["QDRANT"]["url"])
collection_name = config["QDRANT"]["collection_name"]
log = structlog.get_logger()
source_file = config["DOCUMENT_PROCESSING"]["source_file"]
loader = TextLoader(source_file)
docs = loader.load()
embeddings_model = OllamaEmbeddings(
    model=config["EMBEDDING"]["model"], base_url=config["EMBEDDING"]["base_url"]
)
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=config["DOCUMENT_PROCESSING"]["chunk_size"],
    chunk_overlap=config["DOCUMENT_PROCESSING"]["chunk_overlap"],
)
existing_collections = qdrant.get_collections().collections
collection_names = [c.name for c in existing_collections]

if collection_name not in collection_names:
    log.info(f"Collection {collection_name} not found - creating...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
else:
    log.info(f"Using existing collection {collection_name}")
    log.info(f"Clearing all existing points from collection {collection_name}")
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=Filter(must=[])),
    )

chunks = markdown_splitter.split_documents(docs)

log.info(f"Chunks: {len(chunks)}")

created_at = datetime.now(UTC)
texts = [chunk.page_content for chunk in chunks]
embeddings = []
for text in tqdm(texts, desc="Ollama Embeddings"):
    emb = embeddings_model.embed_query(text)
    embeddings.append(emb)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"text": texts[i], "source": source_file, "created_at": created_at},
    )
    for i in range(len(embeddings))
]

qdrant.upsert(collection_name=collection_name, points=points)
