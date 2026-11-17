"""Predict API endpoint for on-demand Gaussian Process microclimate predictions."""

import json
from typing import Any, Dict, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.app.config import DATA_DIR, settings
from backend.model.frost import classify_risk_tier, compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork

router = APIRouter(prefix="/api", tags=["predict"])

# Lazy singletons
_parcels_net: Optional[ParcelNetwork] = None
_sensors_net: Optional[SensorNetwork] = None


def get_components():
    global _parcels_net, _sensors_net
    if _parcels_net is None:
        _parcels_net = ParcelNetwork()
    if _sensors_net is None:
        _sensors_net = SensorNetwork()
    return _parcels_net, _sensors_net


@router.get("/predict")
def predict_parcels(
    night: str = Query(default=settings.DEMO_HERO_NIGHT, description="Night identifier, e.g. night_001"),
    hour: int = Query(default=4, ge=0, le=23, description="Hour of the night, default 4 (04:00 peak frost)"),
    threshold_c: float = Query(default=settings.FROST_THRESHOLD_C, description="Frost temperature threshold in °C"),
) -> Dict[str, Any]:
    """Fit Universal Kriging GP on current active sensors and return mu, sigma, and p_frost for all parcels."""
    parcels_net, sensors_net = get_components()

    # Load sensor temperatures for the given night and hour from replay cache
    replay_file = DATA_DIR / "replay" / f"{night}.json"
    if not replay_file.exists():
        raise HTTPException(status_code=404, detail=f"Night {night} replay not found")

    with open(replay_file, "r", encoding="utf-8") as f:
        replay_data = json.load(f)

    frame = None
    for fr in replay_data["frames"]:
        if fr["hour"] == hour:
            frame = fr
            break
    if frame is None:
        frame = replay_data["frames"][8]  # fallback to 04:00

    # Get active sensor readings
    stations = sensors_net.stations
    s_readings = []
    st_coords = []

    for s in stations:
        val = frame["sensor_readings"].get(s.id, frame["regional_temp"])
        s_readings.append(val)
        st_coords.append([s.x, s.y, s.rel_elev])

    s_readings_arr = np.array(s_readings)
    st_coords_arr = np.array(st_coords)
    parcels_coords_arr = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels_net.parcels])

    # Fit Universal Kriging GP model with frozen hyperparameters
    gp = UniversalKrigingGP()
    gp.fit(st_coords_arr, s_readings_arr)
    mu, sigma = gp.predict(parcels_coords_arr)

    p_frost = compute_frost_probability(mu, sigma, threshold_c=threshold_c)
    critical, watch = classify_risk_tier(p_frost, mu, sigma, threshold_c=threshold_c)

    results = []
    for idx, p in enumerate(parcels_net.parcels):
        results.append({
            "parcel_id": p.id,
            "name": p.name,
            "elevation": p.elevation,
            "rel_elev": p.rel_elev,
            "mu": float(mu[idx]),
            "sigma": float(sigma[idx]),
            "p_frost": round(float(p_frost[idx]), 3),
            "is_critical": bool(critical[idx]),
            "is_watch_tier": bool(watch[idx]),
            "is_frost_pocket": p.is_frost_pocket,
        })

    return {
        "night": night,
        "hour": hour,
        "threshold_c": threshold_c,
        "sensor_count": len(stations),
        "mean_uncertainty_sigma": round(float(np.mean(sigma)), 3),
        "predictions": results,
    }
