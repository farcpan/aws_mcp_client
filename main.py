import asyncio
from datetime import datetime
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import requests
import time

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
            # initialization
            await session.initialize()

            # searching documents
            start = time.time()
            result = await session.call_tool(
                "search_documentation",
                {
                    "search_phrase": search_keyword
                }
            )
            search_results = result.structuredContent["search_results"]

            # status_code, results(or error message)
            if search_results is None or len(search_results) <= 0:
                return 404, "no search results", [], 0, 0

            latency_search_ms = time.time() - start

            # reading documents
            start = time.time()
            first_doc = search_results[0]
            url = first_doc["url"]

            doc = await session.call_tool(
                "read_documentation",
                {
                    "url": url
                }
            )
            if doc.content is None or len(doc.content) <= 0:
                return 404, "no documents", search_results, latency_search_ms, 0

            markdown = doc.content[0].text
            latency_read_ms = time.time() - start

            return 200, markdown, search_results, latency_search_ms, latency_read_ms


# Asking questions to LM Studio Endpoint wrapper
def ask_llm(question, context):
    start = time.time()
    answer = call_llm(
        question=question,
        context=context
    )
    return answer, time.time() - start


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


def save_rag_log(data, output_dir = "logs"):
    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # ensure directory
    os.makedirs(output_dir, exist_ok=True)

    # file path
    file_path = os.path.join(output_dir, f"{timestamp}.json")

    # save json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


if __name__ == "__main__":
    question = "API GatewayのRestAPIが待機できる最大時間は？"
    expected_answer = "29秒、ただし延長は可能。"

    # Document from AWS MCP Server
    status_code, result_doc, search_results, latency_search_ms, latency_read_ms = asyncio.run(ask_aws(search_keyword=question))
    if status_code != 200:
        print(f"statusCode: {status_code}")
        print(result_doc)
        exit(1)


    # Asking LLM about your question with the obtained documents from AWS MCP Server
    answer, latency_llm_sec = ask_llm(question=question, context=result_doc)

    # logs
    rag_log = {
        "question": question,
        "expected_answer": expected_answer,
        "search_results": search_results,
        "context": result_doc,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "latency_search_ms": latency_search_ms,
        "latency_read_ms": latency_read_ms,
        "latency_llm_sec": latency_llm_sec,
    }
    save_rag_log(data=rag_log, output_dir="logs")
