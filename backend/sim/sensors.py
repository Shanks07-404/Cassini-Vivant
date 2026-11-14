"""Weather station network and sensor simulation for Cassini Vivant.

Default setup: 6 stations strategically placed on ridge and mid-to-lower slopes,
strictly OUTSIDE the low-lying valley frost pocket (is_pocket == False).
The lowest station's relative elevation is 12.6m, just above the 8m threshold.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from backend.sim.terrain import SyntheticTerrain


@dataclass
class Station:
    """Ground meteorological weather station."""
    id: str
    name: str
    x: float
    y: float
    lon: float
    lat: float
    elevation: float
    rel_elev: float
    is_pocket: bool
    status: str = "active"
    battery_pct: int = 98

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "elevation": round(self.elevation, 2),
            "rel_elev": round(self.rel_elev, 2),
            "is_pocket": self.is_pocket,
            "status": self.status,
            "battery_pct": self.battery_pct,
            "is_simulated": True,
        }

    def to_feature(self, current_temp: Optional[float] = None) -> Dict[str, Any]:
        props = self.to_dict()
        if current_temp is not None:
            props["temp_c"] = round(current_temp, 2)
        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {
                "type": "Point",
                "coordinates": [self.lon, self.lat],
            },
            "properties": props,
        }


DEFAULT_STATION_CONFIGS = [
    {
        "id": "S01",
        "name": "Station Basse Pente (Bord de Combe)",
        "x": 160.0,
        "y": 200.0,
    },
    {
        "id": "S02",
        "name": "Station Coteau Est (Mi-Pente)",
        "x": 450.0,
        "y": -500.0,
    },
    {
        "id": "S03",
        "name": "Station Coteau Ouest (Mi-Pente)",
        "x": -650.0,
        "y": 300.0,
    },
    {
        "id": "S04",
        "name": "Station Haut Versant Sud",
        "x": 850.0,
        "y": -1200.0,
    },
    {
        "id": "S05",
        "name": "Station Haut Versant Nord",
        "x": -1300.0,
        "y": 100.0,
    },
    {
        "id": "S06",
        "name": "Station Plateau Est",
        "x": 1350.0,
        "y": 650.0,
    },
]


class SensorNetwork:
    """Manager for active weather stations."""

    def __init__(self, terrain: Optional[SyntheticTerrain] = None):
        self.terrain = terrain or SyntheticTerrain()
        self.stations: List[Station] = []
        self._stations_by_id: Dict[str, Station] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.stations = []
        self._stations_by_id = {}
        for cfg in DEFAULT_STATION_CONFIGS:
            feats = self.terrain.get_features(cfg["x"], cfg["y"])
            st = Station(
                id=cfg["id"],
                name=cfg["name"],
                x=cfg["x"],
                y=cfg["y"],
                lon=feats.lon,
                lat=feats.lat,
                elevation=feats.elevation,
                rel_elev=feats.rel_elev,
                is_pocket=feats.is_valley_bottom,
            )
            self.stations.append(st)
            self._stations_by_id[st.id] = st

    def get(self, station_id: str) -> Optional[Station]:
        return self._stations_by_id.get(station_id)

    def add_station(self, lat: float, lon: float, name: Optional[str] = None) -> Station:
        """Add a custom user station (e.g. from map click or CSV)."""
        x, y = self.terrain.to_local(lon, lat)
        feats = self.terrain.get_features(x, y)
        next_idx = len(self.stations) + 1
        st_id = f"S{next_idx:02d}"
        st_name = name or f"Station Utilisateur #{st_id}"

        st = Station(
            id=st_id,
            name=st_name,
            x=x,
            y=y,
            lon=round(lon, 6),
            lat=round(lat, 6),
            elevation=feats.elevation,
            rel_elev=feats.rel_elev,
            is_pocket=feats.is_valley_bottom,
        )
        self.stations.append(st)
        self._stations_by_id[st_id] = st
        return st

    def remove_station(self, station_id: str) -> bool:
        if station_id in self._stations_by_id:
            st = self._stations_by_id.pop(station_id)
            self.stations.remove(st)
            return True
        return False

    def reset_defaults(self) -> None:
        self._init_defaults()

    def to_geojson(self, current_temps: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        features = []
        for st in self.stations:
            temp = current_temps.get(st.id) if current_temps else None
            features.append(st.to_feature(current_temp=temp))
        return {
            "type": "FeatureCollection",
            "metadata": {
                "count": len(self.stations),
                "simulated": True,
            },
            "features": features,
        }

    def get_positions(self) -> np.ndarray:
        """Return (N, 2) array of local metric (x, y) coordinates."""
        return np.array([[s.x, s.y] for s in self.stations])

    def get_covariates(self) -> np.ndarray:
        """Return (N, 3) array of [x, y, rel_elev] for GP regression."""
        return np.array([[s.x, s.y, s.rel_elev] for s in self.stations])
