"""Scripted-model caller tests. No model weights, GPU, or live API required."""

import asyncio
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import parallel_search, web_search


class MCPTransport:
    def __init__(self):
        self.requests = []
        self.failure = None
        self.delay = 0
        self.results = [{"url": "https://example.org/evidence", "title": "Evidence", "excerpts": ["EVIDENCE_MARKER"]}]
        self.closed_clients = []
        self.original_client = httpx.AsyncClient

    async def handle(self, request):
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        self.requests.append((request, message))
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            await asyncio.sleep(self.delay)
            data = {"results": self.results}
            result = {"content": [{"type": "text", "text": self.failure or json.dumps(data)}],
                      "isError": bool(self.failure)}
            if not self.failure:
                result["structuredContent"] = data
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def client(self, **kwargs):
        client = self.original_client(transport=httpx.MockTransport(self.handle), **kwargs)
        self.closed_clients.append(client)
        return client

    @property
    def calls(self):
        return [m["params"] for _, m in self.requests if m["method"] == "tools/call"]


def load_evaluator(path, outputs, prompts):
    """Import the real evaluator; substitute only heavyweight model dependencies."""
    class Inputs(dict):
        input_ids = [[1]]

        def to(self, _):
            return self

    class Model:
        device = "cpu"

        def to(self, _):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2]]

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"][-1]["text"])
            return "scripted input"

        def __call__(self, **kwargs):
            return Inputs(input_ids=[[1]])

        def batch_decode(self, *args, **kwargs):
            value = next(outputs)
            if isinstance(value, BaseException):
                raise value
            return [value]

    modules = {
        "torch": types.SimpleNamespace(bfloat16="bf16", device=lambda rank: "cpu"),
        "tqdm": types.SimpleNamespace(tqdm=lambda items: items),
        "transformers": types.SimpleNamespace(
            Qwen2_5_VLForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: Processor())),
        "qwen_vl_utils": types.SimpleNamespace(process_vision_info=lambda messages: ([], [])),
    }
    spec = importlib.util.spec_from_file_location("scripted_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    questions = [{"id": i, "question": "Find a fact", "answer": "verified", "golden_answers": ["verified"], "image_path": "fixture.jpg"}
                 for i in range(2)]
    dataset = json.dumps(questions) if "MAT-Search-Benchmark" in str(path) else "\n".join(json.dumps(q) for q in questions)
    module.open = lambda *a, **k: io.StringIO(dataset)
    return module


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.transport = MCPTransport()
        self.http = patch.object(parallel_search.httpx, "AsyncClient", self.transport.client)
        self.http.start()

    def tearDown(self):
        self.http.stop()
        self.env.stop()

    def test_all_production_agent_loops_consume_search_and_isolate_questions(self):
        paths = sorted(ROOT.glob("*/*visual_arft.py"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(script=path.name):
                self.transport.requests.clear()
                prompts = []
                outputs = iter(["<search>first fact query</search>", "<search>follow up query</search>",
                                "<answer>verified</answer>"] * 2)
                evaluator = load_evaluator(path, outputs, prompts)
                result = evaluator.run(0, 1)
                self.assertEqual([x["pred_answer"] for x in result], ["verified", "verified"])
                self.assertIn("<information> 1. Content:EVIDENCE_MARKER", prompts[1])
                self.assertEqual(prompts[2].count("EVIDENCE_MARKER"), 2)
                self.assertNotIn("EVIDENCE_MARKER", prompts[3])
                calls = self.transport.calls
                ids = [c["arguments"]["session_id"] for c in calls]
                self.assertEqual(ids[0], ids[1])
                self.assertEqual(ids[2], ids[3])
                self.assertNotEqual(ids[0], ids[2])
                self.assertEqual(calls[0]["name"], "web_search")
                self.assertEqual(calls[0]["arguments"]["search_queries"], ["first fact query"])
                for request, message in self.transport.requests:
                    self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
                    self.assertEqual(request.headers["User-Agent"], parallel_search.USER_AGENT)
                    self.assertNotIn("authorization", request.headers)
        self.assertTrue(all(client.is_closed for client in self.transport.closed_clients))

    def test_configured_choices_override_and_disable(self):
        with patch.dict(os.environ, {"SERPER_API": "fixture-key"}):
            with patch.object(web_search, "web_search_SERPER_API", return_value=[]) as serper:
                web_search.SearchSession().search("one query")
                serper.assert_called_once_with("one query", 4, timeout=30)
            with patch.dict(os.environ, {"VISUAL_ARFT_SEARCH_PROVIDER": "parallel"}):
                self.assertEqual(web_search.SearchSession().provider, "parallel")
                self.assertEqual(web_search.SearchSession(provider="ddg").provider, "ddg")
            with patch.dict(os.environ, {"VISUAL_ARFT_SEARCH_PROVIDER": "disabled"}):
                with self.assertRaisesRegex(RuntimeError, "disabled"):
                    web_search.SearchSession().search("one query")
        self.assertEqual(self.transport.calls, [])
        with patch.object(web_search, "web_search_DDG", return_value=[]) as ddg:
            web_search.SearchSession(provider="ddg", timeout=7).search("one query")
            ddg.assert_called_once_with("one query", 4, timeout=7)
        with self.assertRaisesRegex(ValueError, "SERPER_API"):
            web_search.SearchSession(provider="serper").search("one query")
        with self.assertRaises(ValueError):
            web_search.SearchSession(provider="unknown")

    def test_dotenv_choice_and_environment_precedence(self):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        self.assertEqual(web_search.SearchSession().provider, "parallel")
        os.environ.pop("SERPER_API", None)
        load_dotenv(stream=io.StringIO("SERPER_API=fixture-key\nVISUAL_ARFT_SEARCH_PROVIDER=serper\n"))
        self.assertEqual(web_search.SearchSession().provider, "serper")
        os.environ["VISUAL_ARFT_SEARCH_PROVIDER"] = "disabled"
        load_dotenv(stream=io.StringIO("VISUAL_ARFT_SEARCH_PROVIDER=parallel\n"))
        self.assertEqual(web_search.SearchSession().provider, "disabled")

    def test_interleaved_sessions_and_retries_keep_identity(self):
        first, second = web_search.SearchSession(), web_search.SearchSession()
        first.search("first question")
        second.search("second question")
        first.search("first follow up")
        ids = [c["arguments"]["session_id"] for c in self.transport.calls]
        self.assertEqual(ids, [first.session_id, second.session_id, first.session_id])
        self.assertNotEqual(first.session_id, second.session_id)
        self.transport.failure = "free anonymous quota exceeded"
        with self.assertRaises(Exception):
            first.search("first retry")
        self.transport.failure = None
        first.search("first retry")
        self.assertEqual([c["arguments"]["session_id"] for c in self.transport.calls[-2:]], [first.session_id] * 2)

    def test_errors_timeout_and_interrupt_reach_caller_without_fallback(self):
        path = next(ROOT.glob("MAT*/*visual_arft.py"))
        for failure in ("free anonymous quota exceeded", "provider failed"):
            with self.subTest(failure=failure), patch.object(web_search, "web_search_SERPER_API") as serper:
                self.transport.failure = failure
                evaluator = load_evaluator(path, iter(["<search>one query</search>"] * 2), [])
                self.assertEqual([r["pred_answer"] for r in evaluator.run(0, 1)], [None, None])
                serper.assert_not_called()
        self.transport.failure = None
        self.transport.delay = 1
        with self.assertRaises(TimeoutError):
            web_search.SearchSession(timeout=0.05).search("one query")
        self.assertTrue(all(client.is_closed for client in self.transport.closed_clients))
        evaluator = load_evaluator(path, iter([KeyboardInterrupt()]), [])
        with self.assertRaises(KeyboardInterrupt):
            evaluator.run(0, 1)

    def test_disabled_configured_serper_and_timeout_in_production_loop(self):
        path = next(ROOT.glob("MAT*/*visual_arft.py"))
        with patch.dict(os.environ, {"VISUAL_ARFT_SEARCH_PROVIDER": "disabled"}):
            evaluator = load_evaluator(path, iter(["<search>one query</search>", "<answer>verified</answer>"]), [])
            self.assertEqual([r["pred_answer"] for r in evaluator.run(0, 1)], [None, "verified"])
        self.assertEqual(self.transport.calls, [])
        response = types.SimpleNamespace(
            text=json.dumps({"organic": [{"title": "Evidence", "link": "https://example.org", "snippet": "SERPER_MARKER"}]}),
            raise_for_status=lambda: None)
        with patch.dict(os.environ, {"SERPER_API": "fixture-key"}), patch.object(
            web_search.requests, "request", return_value=response
        ) as request:
            prompts = []
            evaluator = load_evaluator(path, iter(["<search>one query</search>", "<answer>verified</answer>"] * 2), prompts)
            self.assertEqual([r["pred_answer"] for r in evaluator.run(0, 1)], ["verified", "verified"])
            self.assertIn("SERPER_MARKER", prompts[1])
            self.assertEqual(request.call_args.args, ("POST", "https://google.serper.dev/search"))
            self.assertEqual(request.call_args.kwargs["headers"]["X-API-KEY"], "fixture-key")
            self.assertEqual(request.call_args.kwargs["timeout"], 30)
        self.assertEqual(self.transport.calls, [])
        original_session = web_search.SearchSession
        self.transport.delay = 1
        with patch.object(web_search, "SearchSession", lambda: original_session(timeout=0.05)):
            evaluator = load_evaluator(path, iter(["<search>one query</search>"] * 2), [])
            self.assertEqual([r["pred_answer"] for r in evaluator.run(0, 1)], [None, None])
        self.assertTrue(all(client.is_closed for client in self.transport.closed_clients))

    def test_concurrent_agent_runs_have_distinct_conversation_ids(self):
        from concurrent.futures import ThreadPoolExecutor

        path = next(ROOT.glob("MAT*/*visual_arft.py"))
        evaluators = [load_evaluator(path, iter(["<search>one query</search>",
                                               "<search>follow up query</search>",
                                               "<answer>verified</answer>"] * 2), []) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda evaluator: evaluator.run(0, 1), evaluators))
        self.assertEqual([[r["pred_answer"] for r in rows] for rows in results], [["verified"] * 2] * 2)
        from collections import Counter

        counts = Counter(c["arguments"]["session_id"] for c in self.transport.calls)
        self.assertEqual(len(counts), 4)
        self.assertEqual(set(counts.values()), {2})

    def test_cancellation_closes_active_mcp_transport(self):
        self.transport.delay = 10

        async def run():
            task = asyncio.create_task(parallel_search._search("one query", "fixture-conversation", 30))
            while not self.transport.calls:
                await asyncio.sleep(0.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertTrue(all(client.is_closed for client in self.transport.closed_clients))

    def test_result_contract_empty_results_and_invalid_urls(self):
        session = web_search.SearchSession()
        self.assertEqual(session.search("one query", 1),
                         [{"title": "Evidence", "href": "https://example.org/evidence", "body": "EVIDENCE_MARKER"}])
        self.transport.results = []
        self.assertEqual(session.search("one query"), [])
        self.transport.results = [{"url": "file:///local", "excerpts": ["bad"]}]
        with self.assertRaisesRegex(ValueError, "URL"):
            session.search("one query")
        self.transport.results = [{"url": "https://example.org", "excerpts": "bad"}]
        with self.assertRaisesRegex(ValueError, "excerpts"):
            session.search("one query")


if __name__ == "__main__":
    unittest.main()
