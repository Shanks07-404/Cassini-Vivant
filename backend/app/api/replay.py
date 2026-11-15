"""Replay API endpoint providing precomputed hourly microclimate frames."""

import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from backend.app.config import DATA_DIR
from backend.sim.nights import list_night_ids

router = APIRouter(prefix="/api", tags=["replay"])


@router.get("/replay/{night_id}")
def get_replay(night_id: str) -> Dict[str, Any]:
    """Return precomputed replay frames for the requested night."""
    replay_file = DATA_DIR / "replay" / f"{night_id}.json"
    if not replay_file.exists():
        available = list_night_ids()
        raise HTTPException(
            status_code=404,
            detail=f"Replay data for {night_id} not found. Available: {available[:5]}...",
        )

    with open(replay_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@router.get("/nights")
def list_nights() -> Dict[str, Any]:
    """Return list of available night identifiers."""
    nights = list_night_ids()
    return {
        "count": len(nights),
        "hero_night": "night_001",
        "nights": nights,
    }
