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
from config import settings
import argparse
import uuid
import os

qdrant = QdrantClient(url=settings.qdrant_url)
log = structlog.get_logger()
embeddings_model = OllamaEmbeddings(
    model=settings.embedding_model, base_url=settings.embedding_base_url
)
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


def create_or_clear_collection(collection_name: str) -> None:
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


def ingest_version(version: int, source_file: str) -> None:
    collection_name = settings.get_collection_name(version)
    log.info(
        f"Starting ingestion for Angular v{version} into collection '{collection_name}'"
    )

    if not os.path.exists(source_file):
        log.warning(f"Source file {source_file} not found, skipping version {version}")
        return

    create_or_clear_collection(collection_name)

    loader = TextLoader(source_file)
    docs = loader.load()
    chunks = markdown_splitter.split_documents(docs)
    log.info(f"Chunks for v{version}: {len(chunks)}")

    created_at = datetime.now(UTC)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = []
    for text in tqdm(texts, desc=f"Ollama Embeddings v{version}"):
        emb = embeddings_model.embed_query(text)
        embeddings.append(emb)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "uuid": uuid.uuid4(),
                "text": texts[i],
                "source": source_file,
                "angular_version": version,
                "created_at": created_at,
            },
        )
        for i in range(len(embeddings))
    ]

    qdrant.upsert(collection_name=collection_name, points=points)
    log.info(f"Successfully ingested {len(points)} chunks for Angular v{version}")


def ingest_all() -> None:
    for version, source_file in settings.version_sources.items():
        ingest_version(version, source_file)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Angular documentation into Qdrant"
    )
    parser.add_argument(
        "--version",
        type=int,
        choices=list(settings.version_sources.keys()),
        help="Specific Angular version to ingest (default: all)",
    )
    args = parser.parse_args()

    if args.version:
        source_file = settings.get_source_file(args.version)
        if source_file:
            ingest_version(args.version, source_file)
        else:
            log.error(f"No source file configured for Angular v{args.version}")
    else:
        ingest_all()


if __name__ == "__main__":
    main()
