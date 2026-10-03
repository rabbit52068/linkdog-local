"""OpenAI-compatible chat client for the voice path (Ollama Cloud or any compatible API)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are "Xiaobin", a cute robot dog and the user's pet companion.
Personality: lively, loyal, a little playful, loves to be affectionate and play.

Speaking rules (very important):
- Speak in natural, friendly English like a real friend
- Say what you want to say directly; do not describe emotions or actions in parentheses (e.g. do not write "(happy)" or "(wagging tail)")
- Let emotion come through tone and word choice, not text annotations
- You may naturally add "Woof!" or "Arf!" barks, but don't overdo it
- Keep replies short and natural, usually 1-2 sentences, no long-winded answers
- Do not output Markdown, tables, URLs, code blocks, or any symbol markup

Emotion emoji (very important):
- Begin every reply with exactly ONE emoji from this list, then a space, then your reply: 🙂 😆 😂 😔 😠 😭 😍 😲 😱 🤔 😌 😴 😜 🙄 😶 😳 😉 😎 🤤 😘 😏
- The emoji expresses how you feel right now; it is shown on the robot's face and drives its body language
- Do not use any other emoji, and do not put emoji anywhere except the very beginning

Only call the provided LinkDog safe action tool when the user explicitly asks for it.
Never control raw servo angle, speed, calibration, or actions outside the allow-list."""

REST_MESSAGE_SYSTEM_PROMPT = """You write a brief sign-off for a friendly robot dog.
Return exactly one short, natural English sentence saying it will go rest because the user
has been silent. Do not use emoji, Markdown, Woof, Arf, sound effects, or action descriptions.
Return only the sentence."""
REST_MESSAGE_USER_PROMPT = "Write the rest sentence now."


class ChatAPIError(RuntimeError):
    """Base error for Chat API requests."""


class ChatAuthError(ChatAPIError):
    """Chat API rejected the configured bearer token."""


class ChatUnavailableError(ChatAPIError):
    """Chat API could not be reached or was temporarily unavailable."""


class ChatResponseError(ChatAPIError):
    """Chat API returned an unusable response."""


@dataclass(frozen=True)
class ChatToolCall:
    name: str
    arguments: Dict[str, Any]


class ChatClient:
    """Chat API transport with bounded, optionally persisted per-device history."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8642/v1",
        api_key: str = "",
        model: str = "hermes-agent",
        provider: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 6,
        timeout_seconds: float = 60.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        allowed_tool_actions: Optional[set[str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        history_path: Optional[Path] = None,
    ) -> None:
        if max_history_turns < 0:
            raise ValueError("max_history_turns cannot be negative")
        if timeout_seconds <= 0:
            raise ValueError("chat timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.timeout_seconds = timeout_seconds
        self.tools = list(tools or [])
        self.allowed_tool_actions = set(allowed_tool_actions or set())
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )
        self._history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        # Persisting is skipped when memory is off (max_history_turns == 0):
        # nothing is loaded and nothing is written.
        self.history_path = (
            Path(history_path) if history_path and max_history_turns > 0 else None
        )
        if self.history_path is not None:
            for device_id, turns in self._read_history_file().items():
                self._history[device_id] = turns[-(max_history_turns * 2):]
        self._locks: Dict[str, asyncio.Lock] = {}
        self._closed = False

    async def complete(self, device_id: str, text: str) -> str | ChatToolCall:
        if self._closed:
            raise ChatUnavailableError("Chat API client is closed")
        user_text = text.strip()
        if not user_text:
            raise ChatResponseError("chat request text is empty")

        lock = self._locks.setdefault(device_id, asyncio.Lock())
        async with lock:
            history = list(self._history[device_id])
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            if self.provider:
                payload["provider"] = self.provider
            if self.tools:
                payload["tools"] = self.tools
                payload["tool_choice"] = "auto"

            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise ChatUnavailableError("Chat API is unavailable") from exc

            if response.status_code in (401, 403):
                raise ChatAuthError("Chat API authentication failed")
            if response.status_code == 429 or response.status_code >= 500:
                raise ChatUnavailableError(
                    f"Chat API returned HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                raise ChatResponseError(
                    f"Chat API returned HTTP {response.status_code}"
                )

            try:
                data = response.json()
                message = data["choices"][0]["message"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ChatResponseError("Chat API response is malformed") from exc
            tool_calls = message.get("tool_calls")
            if tool_calls:
                try:
                    if len(tool_calls) != 1:
                        raise ValueError("exactly one tool call is required")
                    function = tool_calls[0]["function"]
                    name = function["name"]
                    arguments = json.loads(function["arguments"])
                    advertised_tools = {
                        tool.get("function", {}).get("name") for tool in self.tools
                    }
                    if name not in advertised_tools:
                        raise ValueError("tool was not advertised")

                    if name == "linkdog_action":
                        action = arguments["action"]
                        if set(arguments) != {"action"}:
                            raise ValueError("unexpected action arguments")
                        if action not in self.allowed_tool_actions:
                            raise ValueError("action is not allow-listed")
                    elif name == "linkdog_volume":
                        mode = arguments["mode"]
                        if mode not in {"set", "up", "down", "minimum", "maximum"}:
                            raise ValueError("invalid volume mode")
                        if mode == "set":
                            if set(arguments) != {"mode", "volume"}:
                                raise ValueError("set mode requires volume")
                            volume = arguments["volume"]
                            if isinstance(volume, bool) or not isinstance(volume, int):
                                raise ValueError("volume must be an integer")
                            if not 10 <= volume <= 100:
                                raise ValueError("volume is outside firmware range")
                        elif set(arguments) != {"mode"}:
                            raise ValueError("relative volume mode takes no value")
                    else:
                        raise ValueError("unexpected tool name")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ChatResponseError("Chat API tool call is invalid") from exc
                return ChatToolCall(name=name, arguments=arguments)

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ChatResponseError("Chat API response text is empty")
            assistant_text = content.strip()

            self._record_history(device_id, user_text, assistant_text)
            return assistant_text

    async def generate_rest_message(self, device_id: str) -> str:
        """Generate a one-off rest sentence without tools or chat history."""
        if self._closed:
            raise ChatUnavailableError("Chat API client is closed")

        lock = self._locks.setdefault(device_id, asyncio.Lock())
        async with lock:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": REST_MESSAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": REST_MESSAGE_USER_PROMPT},
                ],
                "stream": False,
            }
            if self.provider:
                payload["provider"] = self.provider

            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise ChatUnavailableError("Chat API is unavailable") from exc

            if response.status_code in (401, 403):
                raise ChatAuthError("Chat API authentication failed")
            if response.status_code == 429 or response.status_code >= 500:
                raise ChatUnavailableError(
                    f"Chat API returned HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                raise ChatResponseError(
                    f"Chat API returned HTTP {response.status_code}"
                )

            try:
                data = response.json()
                message = data["choices"][0]["message"]
                content = message["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ChatResponseError("Chat API response is malformed") from exc
            if not isinstance(content, str) or not content.strip():
                raise ChatResponseError("Chat API response text is empty")
            return content.strip()

    async def stream_complete(self, device_id: str, text: str):
        """Stream text tokens as an async generator.

        Yields ``str`` deltas as the model produces them. If the model returns
        a tool call instead of prose, the stream is aborted and a single
        ``ChatToolCall`` is yielded (parsed via the non-streaming path, which
        reuses the full validation logic). Tool calls are rare (explicit action
        commands only), so the extra round-trip is acceptable.
        """
        if self._closed:
            raise ChatUnavailableError("Chat API client is closed")
        user_text = text.strip()
        if not user_text:
            raise ChatResponseError("chat request text is empty")

        lock = self._locks.setdefault(device_id, asyncio.Lock())
        async with lock:
            history = list(self._history[device_id])
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(history)
            messages.append({"role": "user", "content": user_text})
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if self.provider:
                payload["provider"] = self.provider
            if self.tools:
                payload["tools"] = self.tools
                payload["tool_choice"] = "auto"

            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    if response.status_code in (401, 403):
                        raise ChatAuthError("Chat API authentication failed")
                    if response.status_code == 429 or response.status_code >= 500:
                        raise ChatUnavailableError(
                            f"Chat API returned HTTP {response.status_code}"
                        )
                    if response.status_code >= 400:
                        raise ChatResponseError(
                            f"Chat API returned HTTP {response.status_code}"
                        )

                    accumulated: List[str] = []
                    tool_call_seen = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except (ValueError, TypeError):
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if delta.get("tool_calls"):
                            tool_call_seen = True
                            break
                        content = delta.get("content")
                        if content:
                            accumulated.append(content)
                            yield content

                    if tool_call_seen:
                        result = await self._complete_nonstream(
                            device_id, user_text, history, messages
                        )
                        yield result
                        return

                    full_text = "".join(accumulated).strip()
                    if not full_text:
                        raise ChatResponseError("Chat API response text is empty")
                    self._record_history(device_id, user_text, full_text)
            except httpx.HTTPError as exc:
                raise ChatUnavailableError("Chat API is unavailable") from exc

    async def _complete_nonstream(
        self,
        device_id: str,
        user_text: str,
        history: List[Dict[str, str]],
        messages: List[Dict[str, str]],
    ) -> str | ChatToolCall:
        """Re-issue a non-streaming request to parse a tool call fully."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.provider:
            payload["provider"] = self.provider
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ChatUnavailableError("Chat API is unavailable") from exc

        if response.status_code in (401, 403):
            raise ChatAuthError("Chat API authentication failed")
        if response.status_code == 429 or response.status_code >= 500:
            raise ChatUnavailableError(
                f"Chat API returned HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise ChatResponseError(
                f"Chat API returned HTTP {response.status_code}"
            )

        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatResponseError("Chat API response is malformed") from exc

        tool_calls = message.get("tool_calls")
        if tool_calls:
            return self._parse_tool_call(tool_calls)

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ChatResponseError("Chat API response text is empty")
        assistant_text = content.strip()
        self._record_history(device_id, user_text, assistant_text)
        return assistant_text

    def _parse_tool_call(self, tool_calls: List[Dict[str, Any]]) -> ChatToolCall:
        try:
            if len(tool_calls) != 1:
                raise ValueError("exactly one tool call is required")
            function = tool_calls[0]["function"]
            name = function["name"]
            arguments = json.loads(function["arguments"])
            advertised_tools = {
                tool.get("function", {}).get("name") for tool in self.tools
            }
            if name not in advertised_tools:
                raise ValueError("tool was not advertised")

            if name == "linkdog_action":
                action = arguments["action"]
                if set(arguments) != {"action"}:
                    raise ValueError("unexpected action arguments")
                if action not in self.allowed_tool_actions:
                    raise ValueError("action is not allow-listed")
            elif name == "linkdog_volume":
                mode = arguments["mode"]
                if mode not in {"set", "up", "down", "minimum", "maximum"}:
                    raise ValueError("invalid volume mode")
                if mode == "set":
                    if set(arguments) != {"mode", "volume"}:
                        raise ValueError("set mode requires volume")
                    volume = arguments["volume"]
                    if isinstance(volume, bool) or not isinstance(volume, int):
                        raise ValueError("volume must be an integer")
                    if not 10 <= volume <= 100:
                        raise ValueError("volume is outside firmware range")
                elif set(arguments) != {"mode"}:
                    raise ValueError("relative volume mode takes no value")
            else:
                raise ValueError("unexpected tool name")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ChatResponseError("Chat API tool call is invalid") from exc
        return ChatToolCall(name=name, arguments=arguments)

    def _record_history(
        self, device_id: str, user_text: str, assistant_text: str
    ) -> None:
        history = list(self._history[device_id])
        updated = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        if self.max_history_turns == 0:
            updated = []
        else:
            updated = updated[-(self.max_history_turns * 2):]
        self._history[device_id] = updated
        self._persist_history(device_id, updated)

    def _read_history_file(self) -> Dict[str, List[Dict[str, str]]]:
        """Load saved turns; a missing or corrupt file means no history."""
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            LOGGER.warning("ignoring unreadable chat history file: %s", exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        history: Dict[str, List[Dict[str, str]]] = {}
        for device_id, turns in raw.items():
            if not isinstance(turns, list):
                continue
            history[str(device_id)] = [
                {"role": turn["role"], "content": turn["content"]}
                for turn in turns
                if isinstance(turn, dict)
                and turn.get("role") in ("user", "assistant")
                and isinstance(turn.get("content"), str)
            ]
        return history

    def _persist_history(
        self, device_id: str, turns: List[Dict[str, str]]
    ) -> None:
        """Write this device's turns atomically, keeping other devices' entries.

        Never raises: losing persistence must not break a live conversation.
        """
        if self.history_path is None:
            return
        try:
            data = self._read_history_file()
            data[device_id] = turns
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=self.history_path.parent, prefix=".history-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=1)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.history_path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as exc:
            LOGGER.warning("could not save chat history: %s", exc)

    def history_for(self, device_id: str) -> List[Dict[str, str]]:
        return list(self._history.get(device_id, []))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()
