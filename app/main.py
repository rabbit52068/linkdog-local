"""hermes-linkdog adapter: FastAPI app factory and route wiring.

Routes live in app/routes/; process-wide state lives in app/state.py.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import control, dashboard, device, health, ota

# Timestamps let a turn's stages be lined up; the message itself keeps the
# "[VOICE-ASR] ..." prefixes so existing greps still work.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def _warm_voice_models() -> None:
    """Load ASR and TTS once at startup instead of on the first utterance."""
    for name, build in (("asr", device.build_asr), ("tts", device.build_tts)):
        try:
            warm = getattr(build(), "warm", None)
            if warm is None:
                continue
            warm()
            LOGGER.info(f"[STARTUP] {name} model warmed")
        except Exception as error:  # noqa: BLE001 - warming is best effort
            LOGGER.warning(f"[STARTUP] {name} warm-up failed: {error!r}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_task = None
    if device.voice_input_enabled():
        # Runs in a thread so the server accepts connections immediately.
        warm_task = asyncio.create_task(asyncio.to_thread(_warm_voice_models))
    yield
    if warm_task is not None and not warm_task.done():
        warm_task.cancel()


app = FastAPI(title="hermes-linkdog", lifespan=lifespan)
app.mount(
    "/dashboard/assets",
    StaticFiles(directory=str(dashboard.DASHBOARD_DIR)),
    name="dashboard-assets",
)
for module in (device, control, dashboard, health, ota):
    app.include_router(module.router)
