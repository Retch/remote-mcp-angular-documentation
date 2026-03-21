from fastmcp import Client
import asyncio
import modules.retrieval as Retriever

#retrieved_docs = Retriever.find_docs("ngClass")

#print(len(retrieved_docs))


client = Client("http://localhost:8000/mcp")


async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))
