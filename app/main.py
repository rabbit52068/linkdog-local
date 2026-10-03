"""hermes-linkdog adapter: FastAPI app factory and route wiring.

Routes live in app/routes/; process-wide state lives in app/state.py.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import control, dashboard, device, health, ota

app = FastAPI(title="hermes-linkdog")
app.mount(
    "/dashboard/assets",
    StaticFiles(directory=str(dashboard.DASHBOARD_DIR)),
    name="dashboard-assets",
)
for module in (device, control, dashboard, health, ota):
    app.include_router(module.router)
