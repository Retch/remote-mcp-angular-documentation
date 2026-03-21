from fastmcp import FastMCP
import modules.retrieval as Retriever
from type_definitions.retrieved_doc import RetrievedDoc

mcp = FastMCP("Angular Documentation MCP Server")

@mcp.tool
def search_docs(query: str) -> list[str]:
    retrieved_docs: list[RetrievedDoc] = Retriever.find_docs("ngClass")
    return [t.text for t in retrieved_docs]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
