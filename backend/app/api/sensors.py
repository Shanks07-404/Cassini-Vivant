"""Sensor network API endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.sim.sensors import SensorNetwork

router = APIRouter(prefix="/api", tags=["sensors"])

# Singleton sensor network
_sensors_network: Optional[SensorNetwork] = None


def get_sensors_network() -> SensorNetwork:
    global _sensors_network
    if _sensors_network is None:
        _sensors_network = SensorNetwork()
    return _sensors_network


class AddSensorRequest(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None


@router.get("/sensors")
def list_sensors() -> Dict[str, Any]:
    """Return current active sensor stations as GeoJSON and list."""
    net = get_sensors_network()
    stations_list = [s.to_dict() for s in net.stations]
    geojson = net.to_geojson()
    return {
        "count": len(net.stations),
        "stations": stations_list,
        "geojson": geojson,
    }


@router.post("/sensors")
def add_sensor(req: AddSensorRequest) -> Dict[str, Any]:
    """Add a new sensor station at specified coordinates."""
    net = get_sensors_network()
    st = net.add_station(lat=req.lat, lon=req.lon, name=req.name)
    return {
        "status": "added",
        "station": st.to_dict(),
    }


@router.delete("/sensors/{station_id}")
def delete_sensor(station_id: str) -> Dict[str, Any]:
    """Remove a sensor station by ID."""
    net = get_sensors_network()
    removed = net.remove_station(station_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Sensor {station_id} not found")
    return {
        "status": "deleted",
        "station_id": station_id,
        "remaining_count": len(net.stations),
    }


@router.post("/sensors/reset")
def reset_sensors() -> Dict[str, Any]:
    """Reset to the 6 default reference stations."""
    net = get_sensors_network()
    net.reset_defaults()
    return {
        "status": "reset",
        "count": len(net.stations),
        "stations": [s.to_dict() for s in net.stations],
    }
