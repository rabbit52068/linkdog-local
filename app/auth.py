"""Shared-secret access control for the adapter's control and dashboard APIs.

Device-facing routes (WebSocket, OTA, music) stay open because the firmware
cannot send a token. Everything that can move the dog, read or change
settings, or expose diagnostics goes through :func:`require_token`.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def configured_token() -> str:
    return os.environ.get("LINKDOG_API_TOKEN", "").strip()


async def require_token(request: Request) -> None:
    """Allow loopback callers and LAN callers with the matching bearer token.

    Loopback is always trusted: any local process can already read ``.env``,
    and this keeps the local Hermes MCP bridge free of a copied secret. With
    no token configured, LAN callers are refused outright.
    """
    host = request.client.host if request.client is not None else None
    if host in LOOPBACK_HOSTS:
        return
    token = configured_token()
    if not token:
        raise HTTPException(
            status_code=403,
            detail="LINKDOG_API_TOKEN is not set; API is limited to localhost",
        )

    header = request.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        supplied.strip().encode(), token.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
