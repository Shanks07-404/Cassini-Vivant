"""Next best sensor recommendation, Delaunay mesh, and evaluation API."""

import json
from typing import Any, Dict, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.app.config import DATA_DIR, settings
from backend.model.gp import UniversalKrigingGP
from backend.model.mesh import compute_delaunay_mesh
from backend.model.next_sensor import NextSensorOptimizer
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork

router = APIRouter(prefix="/api", tags=["optimization"])

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


@router.get("/mesh")
def get_mesh() -> Dict[str, Any]:
    """Return Delaunay triangulation network of currently active sensors as GeoJSON."""
    _, sensors_net = get_components()
    return compute_delaunay_mesh(sensors_net)


@router.post("/next-sensor")
def recommend_next_sensor(
    night: str = Query(default=settings.DEMO_HERO_NIGHT),
    hour: int = Query(default=4),
) -> Dict[str, Any]:
    """Find parcel maximizing threshold-weighted variance reduction."""
    parcels_net, sensors_net = get_components()

    # Load sensor temperatures from replay
    replay_file = DATA_DIR / "replay" / f"{night}.json"
    if not replay_file.exists():
        raise HTTPException(status_code=404, detail=f"Night {night} not found")

    with open(replay_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame = [fr for fr in data["frames"] if fr["hour"] == hour][0]
    stations = sensors_net.stations
    s_readings = np.array([
        frame["sensor_readings"].get(s.id, frame["regional_temp"])
        for s in stations
    ])
    st_coords = np.array([
        [s.x, s.y, s.rel_elev]
        for s in stations
    ])

    gp = UniversalKrigingGP()
    gp.fit(st_coords, s_readings)

    optimizer = NextSensorOptimizer(parcels_net)
    recommendation = optimizer.find_next_sensor(gp, sensors_net, s_readings)

    return {
        "status": "success",
        "current_sensor_count": len(stations),
        "recommendation": recommendation,
    }


@router.get("/eval")
def get_evaluation() -> Dict[str, Any]:
    """Return model cross-validation benchmarks and headline next-sensor results."""
    eval_file = DATA_DIR / "eval.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run scripts/run_eval.py first.")

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
