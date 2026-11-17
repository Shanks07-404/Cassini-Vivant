"""Parcels API endpoints."""

import json
from typing import Any, Dict
from fastapi import APIRouter, Query

from backend.app.config import DATA_DIR
from backend.sim.parcels import ParcelNetwork

router = APIRouter(prefix="/api", tags=["parcels"])

_cached_geojson: Dict[str, Any] | None = None
_network: ParcelNetwork | None = None


def get_network() -> ParcelNetwork:
    global _network
    if _network is None:
        _network = ParcelNetwork()
    return _network


def get_parcels_geojson() -> Dict[str, Any]:
    global _cached_geojson
    if _cached_geojson is None:
        cache_file = DATA_DIR / "parcels.geojson"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                _cached_geojson = json.load(f)
        else:
            net = get_network()
            _cached_geojson = net.to_geojson()
            net.save_cache()
    return _cached_geojson


@router.get("/parcels")
def list_parcels(
    placeholder_val: float | None = Query(None, description="Optional override for placeholder risk value (0-1)")
) -> Dict[str, Any]:
    """Return all parcels as a GeoJSON FeatureCollection with terrain properties and risk values."""
    base_geojson = get_parcels_geojson()
    if placeholder_val is None:
        return base_geojson

    # If an override was requested, clone features with modified value
    clamped = max(0.0, min(1.0, placeholder_val))
    cloned_features = []
    for feat in base_geojson["features"]:
        feat_copy = dict(feat)
        props_copy = dict(feat["properties"])
        props_copy["value"] = clamped
        props_copy["risk_score"] = clamped
        feat_copy["properties"] = props_copy
        cloned_features.append(feat_copy)

    return {
        **base_geojson,
        "features": cloned_features,
    }
