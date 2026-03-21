from fastmcp import FastMCP
import modules.retrieval as Retriever
from type_definitions.retrieved_doc import RetrievedDoc
import structlog

log = structlog.get_logger()
mcp = FastMCP("Angular Documentation MCP Server")

@mcp.tool(description="Search angular documentation by query which may be words and sentences.")
def search_docs(query: str) -> list[str]:
    log.info(f"Asking retriever for query: {query}")
    retrieved_docs: list[RetrievedDoc] = Retriever.find_docs(query)
    log.info(f"Retriever found {len(retrieved_docs)} docs")
    return [t.text for t in retrieved_docs]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
