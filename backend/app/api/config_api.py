"""Application configuration and metadata API."""

from typing import Any, Dict
from fastapi import APIRouter

from backend.app.config import settings

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
def get_config() -> Dict[str, Any]:
    """Return runtime configuration, coordinate bounds, thresholds, and honesty flags."""
    return {
        "app_name": "Cassini Vivant",
        "tagline": "La triangulation du terroir, en temps réel",
        "honesty_badge": settings.HONESTY_BADGE,
        "is_simulated_data": settings.IS_SIMULATED_DATA,
        "frost_threshold_c": settings.FROST_THRESHOLD_C,
        "demo_hero_night": settings.DEMO_HERO_NIGHT,
        "demo_parcel_id": settings.DEMO_PARCEL_ID,
        "center": {
            "lat": settings.CENTER_LAT,
            "lon": settings.CENTER_LON,
            "region": "Vallée Synthétique (47.81° N, 3.80° E - Terroir Fictif)",
        },
        "default_language": settings.DEFAULT_LANGUAGE,
        "offline_mode": not bool(settings.GEMINI_API_KEY),
    }
