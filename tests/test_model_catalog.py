import json
import unittest

import httpx

from app.model_catalog import (
    CatalogResult,
    CatalogUpstreamError,
    OllamaModelCatalog,
    filter_highest_version_models,
    parse_model_family_version,
)


class ModelNameFilteringTests(unittest.TestCase):
    def test_only_approved_families_are_admitted(self):
        models = [
            "deepseek-v4-flash:0731",
            "glm-5.3",
            "minimax-m3",
            "nemotron-3-nano:30b",
            "gpt-oss:20b",
            "qwen3.5:cloud",
            "unrelated-1",
        ]

        self.assertEqual(
            filter_highest_version_models(models),
            ["deepseek-v4-flash:0731", "glm-5.3", "minimax-m3"],
        )

    def test_only_highest_numeric_version_per_family_is_retained(self):
        models = [
            "glm-5.1",
            "glm-5.2",
            "glm-5.3",
            "glm-5.3-flash",
            "deepseek-v3",
            "deepseek-v4-flash:0731",
            "deepseek-v4-pro:0813",
            "minimax-m2.7",
            "minimax-m3",
        ]

        self.assertEqual(
            filter_highest_version_models(models),
            [
                "deepseek-v4-flash:0731",
                "deepseek-v4-pro:0813",
                "glm-5.3",
                "glm-5.3-flash",
                "minimax-m3",
            ],
        )

    def test_versions_are_compared_as_integer_tuples(self):
        self.assertEqual(
            filter_highest_version_models(["glm-5.9", "glm-5.10"]),
            ["glm-5.10"],
        )

    def test_unparseable_names_under_allowed_prefix_are_skipped(self):
        self.assertEqual(
            filter_highest_version_models(
                ["deepseek-latest", "glm-latest", "minimax-cloud", "glm-5.3"]
            ),
            ["glm-5.3"],
        )

    def test_output_is_case_insensitive_alphabetical(self):
        self.assertEqual(
            filter_highest_version_models(
                ["MINIMAX-m3", "glm-5.3-flash", "DeepSeek-v4-pro:0813"]
            ),
            ["DeepSeek-v4-pro:0813", "glm-5.3-flash", "MINIMAX-m3"],
        )

    def test_family_and_version_parser_is_anchored(self):
        self.assertEqual(parse_model_family_version("deepseek-v4-flash:0731"), ("deepseek", (4,)))
        self.assertEqual(parse_model_family_version("glm-5.3-flash"), ("glm", (5, 3)))
        self.assertEqual(parse_model_family_version("minimax-m3"), ("minimax", (3,)))
        self.assertIsNone(parse_model_family_version("prefix-glm-99"))


class OllamaModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    def build_catalog(self, handler, *, clock=lambda: 100.0, ttl=60.0):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        return OllamaModelCatalog("secret-key", client=client, clock=clock, cache_ttl_seconds=ttl)

    async def test_fetches_only_approved_names_with_bearer_auth_and_filters_capabilities(self):
        requests = []
        model_ids = [
            "glm-5.2",
            "glm-5.3",
            "glm-5.3-flash",
            "deepseek-v4-flash:0731",
            "minimax-m3",
            "nemotron-3-nano:30b",
            "gpt-oss:20b",
        ]

        def handler(request):
            requests.append(request)
            self.assertEqual(request.headers["authorization"], "Bearer secret-key")
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": model} for model in model_ids]})
            model = json.loads(request.content)["model"]
            capabilities = ["completion", "tools"]
            if model == "glm-5.3-flash":
                capabilities = ["completion"]
            return httpx.Response(200, json={"capabilities": capabilities})

        result = await self.build_catalog(handler).get_models()

        self.assertEqual(
            result,
            CatalogResult(
                models=("deepseek-v4-flash:0731", "glm-5.3", "minimax-m3"),
                stale=False,
            ),
        )
        shown_models = [json.loads(request.content)["model"] for request in requests[1:]]
        self.assertEqual(
            shown_models,
            ["glm-5.2", "glm-5.3", "glm-5.3-flash", "deepseek-v4-flash:0731", "minimax-m3"],
        )

    async def test_missing_capabilities_never_admits_model(self):
        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})
            return httpx.Response(200, json={})

        result = await self.build_catalog(handler).get_models()

        self.assertEqual(result.models, ())

    async def test_one_show_failure_skips_that_model_without_admitting_it(self):
        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [
                    {"id": "glm-5.3"}, {"id": "minimax-m3"}
                ]})
            model = json.loads(request.content)["model"]
            if model == "glm-5.3":
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})

        result = await self.build_catalog(handler).get_models()

        self.assertEqual(result.models, ("minimax-m3",))

    async def test_models_endpoint_http_errors_are_explicit(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                catalog = self.build_catalog(lambda request, s=status: httpx.Response(s))
                with self.assertRaises(CatalogUpstreamError):
                    await catalog.get_models()

    async def test_malformed_models_json_is_explicit(self):
        catalog = self.build_catalog(
            lambda request: httpx.Response(200, content=b"not json")
        )

        with self.assertRaises(CatalogUpstreamError):
            await catalog.get_models()

    async def test_timeout_is_explicit(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        with self.assertRaises(CatalogUpstreamError):
            await self.build_catalog(handler).get_models()

    async def test_all_show_requests_failing_is_explicit(self):
        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})
            return httpx.Response(500)

        with self.assertRaises(CatalogUpstreamError):
            await self.build_catalog(handler).get_models()

    async def test_cache_avoids_refresh_until_ttl_expires(self):
        now = [100.0]
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})

        catalog = self.build_catalog(handler, clock=lambda: now[0], ttl=60.0)
        first = await catalog.get_models()
        now[0] = 159.0
        second = await catalog.get_models()

        self.assertEqual(first, second)
        self.assertEqual(calls.count("/v1/models"), 1)

    async def test_refresh_failure_returns_stale_cache(self):
        now = [100.0]
        failing = [False]

        def handler(request):
            if failing[0]:
                return httpx.Response(500)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})

        catalog = self.build_catalog(handler, clock=lambda: now[0], ttl=60.0)
        await catalog.get_models()
        now[0] = 161.0
        failing[0] = True

        result = await catalog.get_models()

        self.assertEqual(result, CatalogResult(models=("glm-5.3",), stale=True))


if __name__ == "__main__":
    unittest.main()
