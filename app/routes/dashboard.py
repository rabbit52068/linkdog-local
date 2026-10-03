"""Dashboard page and its settings / model-catalog API."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import state
from app.auth import require_token
from app.dashboard_settings import DashboardSettings
from app.model_catalog import CatalogUpstreamError
from app.routes import control

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"

router = APIRouter()


class DashboardSettingsRequest(BaseModel):
    agent_name: str
    system_prompt: str
    model: str
    api_url: str = ""
    memory_enabled: bool
    max_history_turns: int
    user_profile: str = ""
    context_memory: str = ""
    volume: int


@router.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@router.get("/api/models", dependencies=[Depends(require_token)])
async def get_models():
    try:
        result = await state.MODEL_CATALOG.get_models()
    except CatalogUpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "models": sorted(result.models, key=str.casefold),
        "stale": result.stale,
    }


@router.get("/api/settings", dependencies=[Depends(require_token)])
async def get_settings():
    try:
        settings = state.load_dashboard_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "settings": settings.to_dict(),
        "connected_devices": sorted(state.ACTIVE_SESSIONS),
        "connected_device_details": [
            {
                "device_id": device_id,
                "ip_address": getattr(state.ACTIVE_SESSIONS[device_id], "ip_address", None),
            }
            for device_id in sorted(state.ACTIVE_SESSIONS)
        ],
    }


@router.put("/api/settings", dependencies=[Depends(require_token)])
async def update_settings(request: DashboardSettingsRequest):
    try:
        settings = DashboardSettings.from_dict(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        catalog = await state.MODEL_CATALOG.get_models()
    except CatalogUpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if settings.model not in catalog.models:
        raise HTTPException(status_code=422, detail="model is not available")

    state.SETTINGS_STORE.save(settings)

    connected_devices = sorted(state.ACTIVE_SESSIONS)
    volume_results = await asyncio.gather(*(
        control.execute_voice_volume(
            device_id,
            {"mode": "set", "volume": settings.volume},
        )
        for device_id in connected_devices
    ), return_exceptions=True)
    volume_applied_to = [
        device_id
        for device_id, result in zip(connected_devices, volume_results)
        if not isinstance(result, Exception)
    ]
    volume_failed_for = [
        device_id
        for device_id, result in zip(connected_devices, volume_results)
        if isinstance(result, Exception)
    ]
    return {
        "settings": settings.to_dict(),
        "connected_devices": connected_devices,
        "connected_device_details": [
            {
                "device_id": device_id,
                "ip_address": getattr(state.ACTIVE_SESSIONS[device_id], "ip_address", None),
            }
            for device_id in connected_devices
        ],
        "restart_required": bool(connected_devices),
        "volume_applied_to": volume_applied_to,
        "volume_failed_for": volume_failed_for,
    }
