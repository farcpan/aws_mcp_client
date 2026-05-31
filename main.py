import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(search_keyword):
    # for booting MCP Server
    # "uv tool run awslabs.aws-documentation-mcp-server" is automatically executed in main.py
    server_params = StdioServerParameters(
        command="uv",   
        args=[
            "tool",
            "run",
            "awslabs.aws-documentation-mcp-server",
        ],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "search_documentation",
                {
                    "search_phrase": search_keyword
                }
            )
            search_results = result.structuredContent["search_results"]

            # status_code, results(or error message)
            if search_results is None or len(search_results) <= 0:
                return 404, "no search results"

            first_doc = search_results[0]
            url = first_doc["url"]

            doc = await session.call_tool(
                "read_documentation",
                {
                    "url": url
                }
            )
            if doc.content is None or len(doc.content) <= 0:
                return 404, "no documents"

            markdown = doc.content[0].text
            return 200, markdown


if __name__ == "__main__":

    search_keyword = "EC2インスタンスのライフサイクルの概要について知りたい。"
    status_code, result_doc = asyncio.run(main(search_keyword=search_keyword))

    print("=" * 30)
    print(f"statusCode: {status_code}")
    print(result_doc)