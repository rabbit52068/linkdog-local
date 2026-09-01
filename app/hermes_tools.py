from typing import Any, Dict, Optional

import httpx

from app.actions import ACTION_SPECS, build_arguments


class LinkDogToolError(RuntimeError):
    pass


class LinkDogClient:
    def __init__(
        self,
        adapter_url: str,
        device_id: str,
        timeout: float = 12.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.adapter_url = adapter_url.rstrip("/")
        self.device_id = device_id
        self.timeout = timeout
        self.transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.adapter_url,
            timeout=self.timeout,
            transport=self.transport,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text or f"HTTP {response.status_code}"
        if isinstance(body, dict):
            return str(body.get("detail") or body)
        return str(body)

    def execute(self, action: str, **params: Any) -> Dict[str, Any]:
        if action not in ACTION_SPECS:
            raise LinkDogToolError(f"action is disabled: {action}")

        # 依動作型別建構官方 MCP arguments（含參數透傳與 clamp）。
        try:
            arguments = build_arguments(action, **params)
        except ValueError as exc:
            raise LinkDogToolError(str(exc)) from exc

        payload: Dict[str, Any] = {
            "device_id": self.device_id,
            "action": action,
        }
        # 把參數一併送給 adapter（adapter 會再依型別建構 arguments）。
        payload.update(params)

        try:
            with self._client() as client:
                response = client.post("/xiaozhi/action", json=payload)
        except httpx.HTTPError as exc:
            raise LinkDogToolError(f"adapter request failed: {exc}") from exc

        if response.status_code != 200:
            raise LinkDogToolError(self._error_detail(response))
        result = response.json()
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise LinkDogToolError("device did not confirm action completion")
        return result

    def status(self) -> Dict[str, Any]:
        try:
            with self._client() as client:
                response = client.get("/health")
        except httpx.HTTPError as exc:
            raise LinkDogToolError(f"adapter request failed: {exc}") from exc

        if response.status_code != 200:
            raise LinkDogToolError(self._error_detail(response))
        result = response.json()
        connected_devices = result.get("connected_devices", [])
        return {
            "adapter_status": result.get("status", "unknown"),
            "device_id": self.device_id,
            "connected": self.device_id in connected_devices,
        }
