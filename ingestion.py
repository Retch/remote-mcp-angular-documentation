from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import structlog

log = structlog.get_logger()
loader = TextLoader("./llms-full.txt")
docs = loader.load()
embeddings_model = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
)

chunks = markdown_splitter.split_documents(docs)

log.info(f"Chunks: {len(chunks)}")

embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks[:1]]
)

print(embeddings)
