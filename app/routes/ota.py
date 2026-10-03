"""Firmware OTA manifests and binaries, plus the device bootstrap call."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from app import state

FIRMWARE_DIR = Path(__file__).resolve().parent.parent.parent / "firmware"

router = APIRouter()


# Song list: return an empty list so the device does not fetch from linkdog.me
@router.get("/xiaozhi/music/list.json")
async def music_list():
    return JSONResponse({"version": 0, "songs": []})


# Custom 1.8.15 is deployed; return the same version to avoid repeated OTA (force=1 forces reflash/downgrade).
@router.get("/xiaozhi/ota/esp32s3/firmware.json")
async def s3_firmware():
    return JSONResponse({
        "latest": {
            "version": "1.8.15",
            "url": f"http://{state.HOST}:{state.PORT}/xiaozhi/ota/esp32s3/linkdog-s3-ota_1.8.15.bin",
        }
    })


# S3 custom firmware download (kept for later USB or other recovery flows)
@router.get("/xiaozhi/ota/esp32s3/linkdog-s3-ota_1.8.15.bin")
async def s3_firmware_bin():
    return FileResponse(
        FIRMWARE_DIR / "linkdog-s3-ota_1.8.15.bin",
        media_type="application/octet-stream",
        filename="linkdog-s3-ota_1.8.15.bin",
    )


# Stock ota_0 app extracted from the repo's 16MB merged image (embedded version 1.8.12).
@router.get("/xiaozhi/ota/esp32s3/linkdog-s3-stock_1.8.12-ota.bin")
async def s3_stock_firmware_bin():
    return FileResponse(
        FIRMWARE_DIR / "linkdog-s3-stock_1.8.12-ota.bin",
        media_type="application/octet-stream",
        filename="linkdog-s3-stock_1.8.12-ota.bin",
    )


# C3 firmware manifest: report the device's current version (2.0.3) so no C3
# upgrade is triggered. No C3 binary is served, so no download URL is given.
@router.get("/xiaozhi/ota/esp32c3/firmware.json")
async def c3_firmware():
    return JSONResponse({"latest": {"version": "2.0.3"}})


@router.post("/xiaozhi/ota/")
async def ota_bootstrap(request: Request):
    # Device POSTs device JSON; body content is not parsed at this stage, only the device id is logged
    body = await request.body()
    device_id = request.headers.get("device-id", "unknown")
    print(f"[OTA] bootstrap from {device_id}, body={len(body)} bytes")

    # Critical: return only the websocket section, never mqtt (otherwise the device uses MQTT)
    return JSONResponse({
        "websocket": {
            "url": f"ws://{state.HOST}:{state.PORT}/xiaozhi/ws",
            "version": 1,
        }
    })
