"""Synthetic procedural terrain model for viticulture microclimate simulation.

Area: 4 km x 4 km synthetic hillside with a meandering valley floor and stream.
All elevation coordinates and slope/aspect features are derived deterministically.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pyproj

from backend.app.config import settings


@dataclass
class TerrainFeatures:
    """Terrain attributes at a given point or parcel centroid."""
    x: float
    y: float
    lon: float
    lat: float
    elevation: float
    rel_elev: float
    slope_deg: float
    aspect_sin: float
    aspect_cos: float
    is_valley_bottom: bool


class SyntheticTerrain:
    """Procedural Digital Elevation Model (DEM) and microclimate topography.
    
    Coordinates:
      - Local: (x, y) in meters relative to terrain center (0, 0), span [-2000, 2000]
      - Geographic: (lon, lat) in WGS84 (EPSG:4326) centered at 47.81° N, 3.80° E
    """

    def __init__(
        self,
        center_lat: float = settings.CENTER_LAT,
        center_lon: float = settings.CENTER_LON,
        extent_m: float = settings.TERRAIN_EXTENT_M,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.extent_m = extent_m
        self.half_extent = extent_m / 2.0

        # Projections
        proj_string = (
            f"+proj=tmerc +lat_0={self.center_lat} +lon_0={self.center_lon} "
            "+datum=WGS84 +units=m +no_defs"
        )
        self.to_wgs84_transformer = pyproj.Transformer.from_crs(
            proj_string, "EPSG:4326", always_xy=True
        )
        self.to_local_transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", proj_string, always_xy=True
        )

    def to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local (x, y) meters to (lon, lat)."""
        lon, lat = self.to_wgs84_transformer.transform(x, y)
        return float(lon), float(lat)

    def to_local(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert (lon, lat) to local (x, y) meters."""
        x, y = self.to_local_transformer.transform(lon, lat)
        return float(x), float(y)

    def stream_x(self, y: float | np.ndarray) -> float | np.ndarray:
        """X position of the meandering valley stream at coordinate y."""
        return 280.0 * np.sin(2.0 * np.pi * y / 3200.0) - 120.0

    def elevation(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Synthetic elevation in meters ASL.
        
        Features:
          - Valley floor base ~145m sloping gently to ~130m northward.
          - Asymmetric valley profile (steeper south-facing slope, ideal grand cru coteau).
          - Rolling knolls and tributary hollows producing diverse parcel mesoclimates.
        """
        # Valley axis and distance
        sx = self.stream_x(y)
        dx = x - sx

        # Gentle valley gradient along river
        base_river = 142.0 - 0.0035 * y

        # Valley cross-section: U/V shaped trough opening up to plateaus
        # Steeper east slope (dx > 0) representing an asymmetric coteau
        asymmetry = np.where(dx >= 0, 1.15, 0.90)
        dist_scaled = np.abs(dx) / (750.0 * asymmetry)
        valley_rise = 135.0 * (1.0 - np.exp(-(dist_scaled ** 1.8)))

        # Secondary hills & rolling knolls (side hollows)
        side_undulation = (
            22.0 * np.sin(2.0 * np.pi * x / 1700.0) * np.cos(2.0 * np.pi * y / 1500.0)
            + 12.0 * np.cos(2.0 * np.pi * x / 2300.0 + 0.5)
        )

        elev = base_river + valley_rise + side_undulation
        return elev

    def valley_floor_elevation(self, y: float | np.ndarray) -> float | np.ndarray:
        """Local valley bottom elevation at given north-south coordinate y."""
        sx = self.stream_x(y)
        return self.elevation(sx, y)

    def relative_elevation(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Height above the local valley drainage floor in meters.
        
        This is a critical physical driver of nocturnal cold air pooling (katabatic drainage).
        """
        floor_z = self.valley_floor_elevation(y)
        z = self.elevation(x, y)
        rel = np.maximum(0.0, z - floor_z)
        return rel

    def slope_and_aspect(
        self, x: float, y: float, delta: float = 5.0
    ) -> Tuple[float, float, float]:
        """Compute slope (degrees) and aspect (sin, cos) using central differences."""
        z_east = self.elevation(x + delta, y)
        z_west = self.elevation(x - delta, y)
        z_north = self.elevation(x, y + delta)
        z_south = self.elevation(x, y - delta)

        dz_dx = (z_east - z_west) / (2.0 * delta)
        dz_dy = (z_north - z_south) / (2.0 * delta)

        grad_mag = np.hypot(dz_dx, dz_dy)
        slope_rad = np.arctan(grad_mag)
        slope_deg = float(np.degrees(slope_rad))

        # Aspect: direction the slope faces (downhill gradient direction)
        # 0 = North, pi/2 = East, pi = South, 3pi/2 = West
        aspect_rad = float(np.arctan2(-dz_dx, -dz_dy) % (2.0 * np.pi))
        aspect_sin = float(np.sin(aspect_rad))
        aspect_cos = float(np.cos(aspect_rad))

        return slope_deg, aspect_sin, aspect_cos

    def get_features(self, x: float, y: float) -> TerrainFeatures:
        """Full terrain profile at local (x, y)."""
        lon, lat = self.to_wgs84(x, y)
        z = float(self.elevation(x, y))
        rel_z = float(self.relative_elevation(x, y))
        slope_deg, aspect_sin, aspect_cos = self.slope_and_aspect(x, y)
        is_valley = rel_z < 8.0  # Cold pool danger zone

        return TerrainFeatures(
            x=x,
            y=y,
            lon=lon,
            lat=lat,
            elevation=round(z, 2),
            rel_elev=round(rel_z, 2),
            slope_deg=round(slope_deg, 2),
            aspect_sin=round(aspect_sin, 4),
            aspect_cos=round(aspect_cos, 4),
            is_valley_bottom=is_valley,
        )
