"""Live Ollama Cloud model discovery and deterministic filtering."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import httpx

LOGGER = logging.getLogger(__name__)
CATALOG_CACHE_TTL_SECONDS = 300.0
OLLAMA_BASE_URL = "https://ollama.com"
_REQUIRED_CAPABILITIES = {"completion", "tools"}

_FAMILY_PATTERNS = {
    "deepseek": re.compile(r"^deepseek-v(\d+(?:\.\d+)*)", re.IGNORECASE),
    "glm": re.compile(r"^glm-(\d+(?:\.\d+)*)", re.IGNORECASE),
    "minimax": re.compile(r"^minimax-m(\d+(?:\.\d+)*)", re.IGNORECASE),
}


class CatalogUpstreamError(RuntimeError):
    """Ollama Cloud could not provide a trustworthy catalog result."""


@dataclass(frozen=True)
class CatalogResult:
    models: tuple[str, ...]
    stale: bool = False


def parse_model_family_version(model_id: str) -> Optional[Tuple[str, tuple[int, ...]]]:
    """Return the approved family and numeric version for one model ID."""
    for family, pattern in _FAMILY_PATTERNS.items():
        match = pattern.match(model_id)
        if match:
            return family, tuple(int(part) for part in match.group(1).split("."))
    return None


def filter_highest_version_models(model_ids: Iterable[str]) -> list[str]:
    """Keep only highest-version variants in approved model families."""
    parsed_by_family: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for model_id in model_ids:
        parsed = parse_model_family_version(model_id)
        if parsed is None:
            if model_id.lower().startswith(tuple(_FAMILY_PATTERNS)):
                LOGGER.warning("Skipping unparseable approved-family model ID: %s", model_id)
            continue
        family, version = parsed
        parsed_by_family[family].append((model_id, version))

    selected: list[str] = []
    for models in parsed_by_family.values():
        highest_version = max(version for _, version in models)
        selected.extend(model_id for model_id, version in models if version == highest_version)
    return sorted(selected, key=str.casefold)


class OllamaModelCatalog:
    """Fetch and cache tool-capable models from Ollama Cloud."""

    def __init__(
        self,
        api_key: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        clock: Callable[[], float] = time.monotonic,
        cache_ttl_seconds: float = CATALOG_CACHE_TTL_SECONDS,
    ) -> None:
        self.api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._clock = clock
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cached_models: Optional[tuple[str, ...]] = None
        self._cached_at = 0.0
        self._refresh_lock = asyncio.Lock()

    async def get_models(self) -> CatalogResult:
        now = self._clock()
        if self._cached_models is not None and now - self._cached_at < self._cache_ttl_seconds:
            return CatalogResult(self._cached_models)

        async with self._refresh_lock:
            now = self._clock()
            if self._cached_models is not None and now - self._cached_at < self._cache_ttl_seconds:
                return CatalogResult(self._cached_models)
            try:
                models = await self._refresh()
            except CatalogUpstreamError:
                if self._cached_models is not None:
                    LOGGER.warning("Ollama model refresh failed; serving stale cache", exc_info=True)
                    return CatalogResult(self._cached_models, stale=True)
                raise
            self._cached_models = tuple(models)
            self._cached_at = now
            return CatalogResult(self._cached_models)

    async def _refresh(self) -> list[str]:
        payload = await self._request_json("GET", "/v1/models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise CatalogUpstreamError("Ollama /v1/models returned malformed data")

        candidates: list[str] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            model_id = item["id"]
            if parse_model_family_version(model_id) is not None:
                candidates.append(model_id)
            elif model_id.lower().startswith(tuple(_FAMILY_PATTERNS)):
                LOGGER.warning("Skipping unparseable approved-family model ID: %s", model_id)

        eligible: list[str] = []
        failed_lookups = 0
        for model_id in candidates:
            try:
                details = await self._request_json(
                    "POST", "/api/show", json_body={"model": model_id}
                )
            except CatalogUpstreamError as exc:
                failed_lookups += 1
                LOGGER.warning("Ollama capability lookup failed for %s: %s", model_id, exc)
                continue
            capabilities = details.get("capabilities")
            if isinstance(capabilities, list) and _REQUIRED_CAPABILITIES.issubset(
                {value for value in capabilities if isinstance(value, str)}
            ):
                eligible.append(model_id)

        if candidates and failed_lookups == len(candidates):
            raise CatalogUpstreamError("all Ollama capability lookups failed")
        return filter_highest_version_models(eligible)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise CatalogUpstreamError("Ollama API key is not configured")
        try:
            response = await self._client.request(
                method,
                OLLAMA_BASE_URL + path,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=json_body,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CatalogUpstreamError(f"Ollama {path} request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise CatalogUpstreamError(f"Ollama {path} returned a non-object response")
        return payload
