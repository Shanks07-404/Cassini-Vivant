"""Synthetic parcel generator and cadastre representation for Cassini Vivant.

Produces ~250 realistic irregular vineyard parcels covering slopes and valley floor.
All geometries are deterministic and derived from a fixed random seed.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform

from backend.app.config import DATA_DIR, settings
from backend.sim.terrain import SyntheticTerrain


FICTIONAL_SECTORS = [
    "Coteau Nord",
    "Combe Ouest",
    "Haut des Vignes",
    "Pente Sud",
    "Vallon de l'Est",
    "Coteau des Roches",
    "Plaine Basse",
    "Versant Ensoleillé",
    "Terrasse Haute",
    "Chemin du Moulin",
    "Clos Fictif",
    "Vignes du Ruisseau",
    "Plateau Venté",
    "Combe Centrale",
    "Coteau Alpha",
    "Bas de Pente",
    "Coteau Beta",
    "Versant Sud-Est",
    "Hauts du Talweg",
    "Parcelle des Pierres",
]


@dataclass
class Parcel:
    """Individual vineyard plot entity."""
    id: str
    name: str
    centroid_x: float
    centroid_y: float
    centroid_lon: float
    centroid_lat: float
    elevation: float
    rel_elev: float
    slope_deg: float
    aspect_sin: float
    aspect_cos: float
    is_valley_bottom: bool
    is_frost_pocket: bool
    area_ha: float
    polygon_wgs84: List[List[float]]  # List of [lon, lat] points
    polygon_local: Polygon

    def to_feature(self, placeholder_risk: float = 0.0) -> Dict[str, Any]:
        """Convert to GeoJSON Feature with properties."""
        # Realistic placeholder value: parcels in low elevation / valley have higher frost risk
        rel_factor = max(0.0, min(1.0, 1.0 - (self.rel_elev / 45.0)))
        val = round(placeholder_risk if placeholder_risk > 0.0 else rel_factor, 3)

        return {
            "type": "Feature",
            "id": self.id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.polygon_wgs84],
            },
            "properties": {
                "id": self.id,
                "name": self.name,
                "centroid_lon": self.centroid_lon,
                "centroid_lat": self.centroid_lat,
                "elevation": self.elevation,
                "rel_elev": self.rel_elev,
                "slope_deg": self.slope_deg,
                "aspect_sin": self.aspect_sin,
                "aspect_cos": self.aspect_cos,
                "is_valley_bottom": self.is_valley_bottom,
                "is_frost_pocket": self.is_valley_bottom,
                "area_ha": self.area_ha,
                "value": val,
                "risk_score": val,
                "is_simulated": True,
            },
        }


class ParcelNetwork:
    """Network of synthetic parcels generated over the terrain."""

    def __init__(self, terrain: Optional[SyntheticTerrain] = None, seed: int = settings.SIMULATION_SEED):
        self.terrain = terrain or SyntheticTerrain()
        self.seed = seed
        self.parcels: List[Parcel] = []
        self._parcels_by_id: Dict[str, Parcel] = {}
        self._generate()

    def _generate(self) -> None:
        """Procedurally generate irregular vine parcels via perturbed Voronoi cells."""
        rng = np.random.RandomState(self.seed)

        # 16 x 16 jittered grid within [-1800m, 1800m]
        grid_dim = 16
        xs = np.linspace(-1800.0, 1800.0, grid_dim)
        ys = np.linspace(-1800.0, 1800.0, grid_dim)
        gx, gy = np.meshgrid(xs, ys)

        jitter_x = (rng.rand(*gx.shape) - 0.5) * 160.0
        jitter_y = (rng.rand(*gy.shape) - 0.5) * 160.0
        points = np.column_stack([(gx + jitter_x).ravel(), (gy + jitter_y).ravel()])

        # Voronoi tessellation
        vor = Voronoi(points)
        domain_box = box(-1900.0, -1900.0, 1900.0, 1900.0)

        raw_polygons: List[Polygon] = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            poly_coords = [vor.vertices[i] for i in region]
            poly = Polygon(poly_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            clipped = poly.intersection(domain_box)
            if clipped.is_valid and not clipped.is_empty and clipped.area > 5000.0:
                raw_polygons.append(clipped)

        # Sort spatially (South to North, West to East) for consistent numbering
        raw_polygons.sort(key=lambda p: (p.centroid.y, p.centroid.x))

        self.parcels = []
        self._parcels_by_id = {}

        for idx, poly in enumerate(raw_polygons, start=1):
            parcel_id = f"P{idx:03d}"
            sector = FICTIONAL_SECTORS[(idx - 1) % len(FICTIONAL_SECTORS)]
            name = f"Parcelle {parcel_id} ({sector})"

            cx = float(poly.centroid.x)
            cy = float(poly.centroid.y)
            lon_c, lat_c = self.terrain.to_wgs84(cx, cy)

            # Sample terrain features across polygon
            features = self.terrain.get_features(cx, cy)
            area_ha = round(float(poly.area) / 10000.0, 2)

            # Convert exterior coordinates to WGS84
            coords_wgs84: List[List[float]] = []
            for px, py in poly.exterior.coords:
                plon, plat = self.terrain.to_wgs84(px, py)
                coords_wgs84.append([round(plon, 6), round(plat, 6)])

            is_pocket = bool(features.rel_elev < 8.0)

            p = Parcel(
                id=parcel_id,
                name=name,
                centroid_x=cx,
                centroid_y=cy,
                centroid_lon=round(lon_c, 6),
                centroid_lat=round(lat_c, 6),
                elevation=features.elevation,
                rel_elev=features.rel_elev,
                slope_deg=features.slope_deg,
                aspect_sin=features.aspect_sin,
                aspect_cos=features.aspect_cos,
                is_valley_bottom=is_pocket,
                is_frost_pocket=is_pocket,
                area_ha=area_ha,
                polygon_wgs84=coords_wgs84,
                polygon_local=poly,
            )
            self.parcels.append(p)
            self._parcels_by_id[parcel_id] = p

    def get(self, parcel_id: str) -> Optional[Parcel]:
        return self._parcels_by_id.get(parcel_id)

    def __len__(self) -> int:
        return len(self.parcels)

    def to_geojson(self) -> Dict[str, Any]:
        """Return GeoJSON FeatureCollection of all parcels."""
        return {
            "type": "FeatureCollection",
            "metadata": {
                "count": len(self.parcels),
                "crs": "EPSG:4326",
                "simulated": True,
                "disclaimer": settings.HONESTY_BADGE,
            },
            "features": [p.to_feature() for p in self.parcels],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return pandas DataFrame of parcel properties for modeling."""
        records = [
            {
                "parcel_id": p.id,
                "name": p.name,
                "x": p.centroid_x,
                "y": p.centroid_y,
                "lon": p.centroid_lon,
                "lat": p.centroid_lat,
                "elevation": p.elevation,
                "rel_elev": p.rel_elev,
                "slope_deg": p.slope_deg,
                "aspect_sin": p.aspect_sin,
                "aspect_cos": p.aspect_cos,
                "is_valley_bottom": p.is_valley_bottom,
                "area_ha": p.area_ha,
            }
            for p in self.parcels
        ]
        return pd.DataFrame(records)

    def save_cache(self, path: Optional[Path] = None) -> Path:
        """Persist GeoJSON cache file for fast offline serving."""
        target = path or (DATA_DIR / "parcels.geojson")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_geojson(), f, indent=2)
        return target
