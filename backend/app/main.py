"""FastAPI entry point for Cassini Vivant."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.config_api import router as config_router
from backend.app.api.parcels import router as parcels_router
from backend.app.api.sensors import router as sensors_router
from backend.app.api.replay import router as replay_router
from backend.app.api.predict import router as predict_router
from backend.app.api.next_sensor_api import router as next_sensor_router
from backend.app.api.roi_api import router as roi_router
from backend.app.api.assistant_api import router as assistant_router

app = FastAPI(
    title="Cassini Vivant API",
    description="Microclimate triangulation & appellation risk intelligence engine for viticulture.",
    version="0.1.0",
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)
app.include_router(parcels_router)
app.include_router(sensors_router)
app.include_router(replay_router)
app.include_router(predict_router)
app.include_router(next_sensor_router)
app.include_router(roi_router)
app.include_router(assistant_router)


@app.get("/")
def root():
    return {
        "status": "online",
        "app": "Cassini Vivant",
        "badge": settings.HONESTY_BADGE,
        "docs": "/docs",
    }
