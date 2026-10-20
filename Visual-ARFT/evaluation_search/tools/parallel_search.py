"""Anonymous Parallel Search MCP adapter for the evaluation agent."""

import asyncio
import json
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Visual-ARFT/0.1 (+https://github.com/Liuziyu77/Visual-RFT)"


async def _search(query, session_id, timeout):
    # Anonymous endpoint: no Parallel credentials or authentication headers.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=timeout
    ) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (reader, writer, _):
            async with ClientSession(
                reader,
                writer,
                read_timeout_seconds=timedelta(seconds=timeout),
                client_info=Implementation(name="Visual-ARFT", version="0.1"),
            ) as client:
                await client.initialize()
                tools = await client.list_tools()
                if not any(tool.name == "web_search" for tool in tools.tools):
                    raise RuntimeError("Parallel MCP does not expose web_search")
                response = await client.call_tool(
                    "web_search",
                    {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": session_id,
                    },
                )
                if response.isError:
                    detail = "\n".join(block.text for block in response.content if block.type == "text")
                    raise RuntimeError("Parallel search failed: " + detail)
                data = response.structuredContent
                if data is None:
                    data = json.loads("\n".join(block.text for block in response.content if block.type == "text"))
                if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                    raise ValueError("Parallel search returned an invalid result list")
                return data["results"]


def search(query, search_num, session_id, timeout):
    """Return the title/href/body shape consumed by the existing agent loop.

    Each call owns its transport; session_id comes from the question's
    SearchSession and survives reconnects. Bound discovery and search together.
    """

    async def run():
        return await asyncio.wait_for(_search(query, session_id, timeout), timeout=timeout)

    items = asyncio.run(run())
    results = []
    for item in items[:search_num]:
        url = item.get("url", "")
        excerpts = item.get("excerpts")
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
            raise ValueError("Parallel search returned an invalid URL")
        if not isinstance(excerpts, list) or any(not isinstance(text, str) for text in excerpts):
            raise ValueError("Parallel search returned invalid excerpts")
        results.append({"title": item.get("title") or "", "href": url, "body": "\n".join(excerpts)})
    return results
