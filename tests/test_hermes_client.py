import json
import unittest

import httpx

from app.hermes_client import (
    DEFAULT_SYSTEM_PROMPT,
    HermesAPIClient,
    HermesAuthError,
    HermesResponseError,
    HermesToolCall,
    HermesUnavailableError,
)


class HermesAPIClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        client = getattr(self, "client", None)
        if client is not None:
            await client.close()

    def make_client(self, handler, max_history_turns=3):
        self.client = HermesAPIClient(
            base_url="http://hermes.test:8642/v1",
            api_key="test-secret",
            system_prompt="Respond briefly in Traditional Chinese.",
            max_history_turns=max_history_turns,
            transport=httpx.MockTransport(handler),
        )
        return self.client

    async def test_sends_openai_request_and_records_successful_turn(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-secret")
            turn = len(requests)
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {"role": "assistant", "content": f"回答{turn}"}
                    }]
                },
            )

        client = self.make_client(handler)

        first = await client.complete("DOG:A", "你好")
        second = await client.complete("DOG:A", "再說一次")

        self.assertEqual(first, "回答1")
        self.assertEqual(second, "回答2")
        self.assertEqual(requests[0]["model"], "hermes-agent")
        self.assertFalse(requests[0]["stream"])
        self.assertEqual(requests[0]["messages"], [
            {"role": "system", "content": "Respond briefly in Traditional Chinese."},
            {"role": "user", "content": "你好"},
        ])
        self.assertEqual(requests[1]["messages"], [
            {"role": "system", "content": "Respond briefly in Traditional Chinese."},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "再說一次"},
        ])


    async def test_history_is_isolated_per_device_and_bounded_by_turns(self):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload["messages"])
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "好"}}]
            })

        client = self.make_client(handler, max_history_turns=1)
        await client.complete("DOG:A", "A1")
        await client.complete("DOG:A", "A2")
        await client.complete("DOG:A", "A3")
        await client.complete("DOG:B", "B1")

        self.assertEqual(requests[2], [
            {"role": "system", "content": "Respond briefly in Traditional Chinese."},
            {"role": "user", "content": "A2"},
            {"role": "assistant", "content": "好"},
            {"role": "user", "content": "A3"},
        ])
        self.assertEqual(requests[3], [
            {"role": "system", "content": "Respond briefly in Traditional Chinese."},
            {"role": "user", "content": "B1"},
        ])

    async def test_includes_explicit_provider_route(self):
        client = HermesAPIClient(
            base_url="http://hermes.test:8642/v1",
            api_key="test-secret",
            model="deepseek-v4-pro",
            provider="ollama-cloud",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.loads(request.content)["provider"]}}]},
                )
            ),
        )
        self.client = client

        result = await client.complete("DOG:A", "你好")

        self.assertEqual(result, "ollama-cloud")

    async def test_returns_validated_tool_call_when_tools_are_configured(self):
        requests = []
        tools = [{
            "type": "function",
            "function": {
                "name": "linkdog_action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["sit_down", "stand_up", "get_down", "shake_hands"],
                        }
                    },
                    "required": ["action"],
                },
            },
        }]

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "linkdog_action",
                            "arguments": '{"action":"sit_down"}',
                        },
                    }],
                }}]
            })

        self.client = HermesAPIClient(
            base_url="http://hermes.test:8642/v1",
            api_key="test-secret",
            tools=tools,
            allowed_tool_actions={"sit_down", "stand_up", "get_down", "shake_hands"},
            transport=httpx.MockTransport(handler),
        )

        result = await self.client.complete("DOG:A", "坐下")

        self.assertEqual(result, HermesToolCall("linkdog_action", {"action": "sit_down"}))
        self.assertEqual(requests[0]["tools"], tools)
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(self.client.history_for("DOG:A"), [])

    async def test_returns_validated_volume_tool_call(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "linkdog_volume",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["set", "up", "down", "minimum", "maximum"],
                        },
                        "volume": {"type": "integer", "minimum": 10, "maximum": 100},
                    },
                    "required": ["mode"],
                },
            },
        }]

        def handler(_request):
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-volume",
                        "type": "function",
                        "function": {
                            "name": "linkdog_volume",
                            "arguments": '{"mode":"set","volume":55}',
                        },
                    }],
                }}]
            })

        self.client = HermesAPIClient(
            base_url="http://hermes.test:8642/v1",
            api_key="test-secret",
            tools=tools,
            transport=httpx.MockTransport(handler),
        )

        result = await self.client.complete("DOG:A", "set volume to 55")

        self.assertEqual(result, HermesToolCall(
            "linkdog_volume", {"mode": "set", "volume": 55}
        ))

    async def test_auth_failure_has_specific_error(self):
        client = self.make_client(lambda _request: httpx.Response(401))

        with self.assertRaises(HermesAuthError):
            await client.complete("DOG:A", "你好")

    async def test_server_and_connection_failures_are_unavailable(self):
        server_client = self.make_client(lambda _request: httpx.Response(503))
        with self.assertRaises(HermesUnavailableError):
            await server_client.complete("DOG:A", "你好")
        await server_client.close()

        def disconnected(request):
            raise httpx.ConnectError("offline", request=request)

        self.client = HermesAPIClient(
            base_url="http://hermes.test:8642/v1",
            api_key="test-secret",
            transport=httpx.MockTransport(disconnected),
        )
        with self.assertRaises(HermesUnavailableError):
            await self.client.complete("DOG:A", "你好")

    async def test_rejects_empty_or_malformed_response_without_saving_turn(self):
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, json={"choices": []})
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "成功"}}]
            })

        client = self.make_client(handler)
        with self.assertRaises(HermesResponseError):
            await client.complete("DOG:A", "失敗這輪")
        result = await client.complete("DOG:A", "新的一輪")

        self.assertEqual(result, "成功")
        self.assertEqual(client.history_for("DOG:A"), [
            {"role": "user", "content": "新的一輪"},
            {"role": "assistant", "content": "成功"},
        ])

    async def test_generate_rest_message_is_short_english_one_shot_without_history_or_tools(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "If there's nothing else, I'll rest now."
                }}]
            })

        client = self.make_client(handler)
        await client.complete("DOG:A", "remember this")
        history_before = client.history_for("DOG:A")

        result = await client.generate_rest_message("DOG:A")

        self.assertEqual(result, "If there's nothing else, I'll rest now.")
        self.assertEqual(client.history_for("DOG:A"), history_before)
        rest_payload = requests[-1]
        self.assertFalse(rest_payload["stream"])
        self.assertNotIn("tools", rest_payload)
        self.assertEqual(len(rest_payload["messages"]), 2)
        prompt_text = " ".join(
            message["content"] for message in rest_payload["messages"]
        ).lower()
        self.assertIn("short", prompt_text)
        self.assertIn("english", prompt_text)
        self.assertIn("do not use", prompt_text)
        self.assertIn("woof", prompt_text)
        self.assertNotIn("remember this", prompt_text)

    def test_default_system_prompt_instructs_emotion_emoji_prefix(self):
        self.assertIn("emoji", DEFAULT_SYSTEM_PROMPT.lower())
        self.assertIn("beginning", DEFAULT_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
