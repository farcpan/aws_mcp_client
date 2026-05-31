import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests

# LM Studio endpoint url (for Gemma4)
LM_STUDIO_ENDPOINT_URL = "http://localhost:12345/v1/chat/completions"


# Wrapper for AWS MCP Server via MCP Client
async def ask_aws(search_keyword):
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


# Asking questions to LM Studio Endpoint wrapper
def ask_llm(question, context):
    answer = call_llm(
        question=question,
        context=context
    )
    return answer


# Wrapper for calling LM Studio Endpoint
def call_llm(question, context):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer using only the provided context."
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]

    payload = {
        "model": "google/gemma-4-e4b",
        "messages": messages,
        "temperature": 0.1
    }

    res = requests.post(LM_STUDIO_ENDPOINT_URL, json=payload)
    res.raise_for_status()

    return res.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    search_keyword = "API GatewayのRestAPIが待機できる最大時間を教えて。"

    # Document from AWS MCP Server
    status_code, result_doc = asyncio.run(ask_aws(search_keyword=search_keyword))
    if status_code != 200:
        print(f"statusCode: {status_code}")
        print(result_doc)
        exit(1)

    # Asking LLM about your question with the obtained documents from AWS MCP Server
    answer = ask_llm(question=search_keyword, context=result_doc)

    print("===== Obtained Documents =====")
    print(result_doc)
    print("===== Final Result =====")
    print(answer)
