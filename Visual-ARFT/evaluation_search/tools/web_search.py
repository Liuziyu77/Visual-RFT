import os
import json
import requests
from typing import List, Literal, Optional, Union
from dotenv import load_dotenv
load_dotenv()

# 使用免费的 duck duck go 进行网页检索，使用 firecrawl 将网页转化为markdown格式
def web_search_DDG(query: Optional[str], search_num: int = 2, search_mode: str = 'fast', *, timeout=30) -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    from duckduckgo_search import DDGS
    if search_mode == 'fast':
        assert type(query)==str
        results = DDGS(timeout=timeout).text(query, max_results=search_num)
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        from firecrawl import FirecrawlApp
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        results = DDGS(timeout=timeout).text(query, max_results=search_num)
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results

def web_search_SERPER_API(query: Optional[str], search_num=2, search_mode: str = 'fast', *, timeout=30) -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    if search_mode == 'fast':
        assert type(query)==str
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload, timeout=timeout)
        response.raise_for_status()
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        from firecrawl import FirecrawlApp
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload, timeout=timeout)
        response.raise_for_status()
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results


class SearchSession:
    """One evaluation question, with a stable MCP identity across its turns.

    An explicit provider wins over the environment. With neither configured,
    retain Serper when its key is present; otherwise use anonymous Parallel.
    Errors never switch providers.
    """

    def __init__(self, provider=None, timeout=30):
        from uuid import uuid4

        configured = provider if provider is not None else os.getenv("VISUAL_ARFT_SEARCH_PROVIDER")
        if configured is None:
            serper_key = os.getenv("SERPER_API", "").strip()
            # The tracked .env ships this dummy value; it is not a configured key.
            configured = "serper" if serper_key and serper_key != "x" * 25 else "parallel"
        self.provider = configured.strip().lower()
        if self.provider not in ("parallel", "serper", "ddg", "disabled"):
            raise ValueError("Search provider must be parallel, serper, ddg, or disabled")
        if timeout <= 0:
            raise ValueError("Search timeout must be positive")
        self.timeout = timeout
        self.session_id = str(uuid4())

    def search(self, query, search_num=4):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be nonempty text")
        if not isinstance(search_num, int) or isinstance(search_num, bool) or search_num < 1:
            raise ValueError("Search result count must be a positive integer")
        if self.provider == "disabled":
            raise RuntimeError("Web search is disabled")
        if self.provider == "parallel":
            from .parallel_search import search

            return search(query, search_num, self.session_id, self.timeout)
        if self.provider == "serper":
            if not os.getenv("SERPER_API", "").strip():
                raise ValueError("Serper requires SERPER_API")
            return web_search_SERPER_API(query, search_num, timeout=self.timeout)
        return web_search_DDG(query, search_num, timeout=self.timeout)
