from fastmcp import FastMCP
import structlog
from typing import Annotated, Optional
from config import settings
from starlette.requests import Request
from starlette.responses import JSONResponse
from modules import retrieval as Retriever
from type_definitions.retrieved_doc import RetrievedDoc

log = structlog.get_logger()
mcp = FastMCP("Angular Documentation MCP Server")


@mcp.tool(
    description="Search angular documentation by query which may be words and sentences."
    f"Available Angular versions: {list(settings.version_sources.keys())}"
)
def search_docs(
                    query: Annotated[str, "Search Angular documentation by keyword, question or example code."],
                    angular_major_version: Annotated[Optional[int], f"Leave empty or use one of these versions: {list(settings.version_sources.keys())}. Default version: {settings.default_version}"] = settings.default_version
                ) -> list[str]:
    log.info(
        f"Asking retriever for query: {query}, Angular v{angular_major_version}")

    if angular_major_version not in settings.version_sources:
        return [
            f"Angular version {angular_major_version} is not configured. Available versions: {list(settings.version_sources.keys())}"
        ]

    retrieved_docs: list[RetrievedDoc] = Retriever.find_docs(
        query, angular_major_version)
    log.info(f"Retriever found {len(retrieved_docs)} docs")
    return [t.text for t in retrieved_docs]


@mcp.custom_route("/metrics", methods=["GET"])
async def metrics(request: Request):
    return JSONResponse(Retriever.get_all_metrics())


if __name__ == "__main__":
    mcp.run(transport="http", port=settings.fastmcp_port)
