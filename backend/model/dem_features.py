"""Standard DEM-derived topographic covariates (TPI, Slope, Aspect, Convergence).

Derived purely from the Digital Elevation Model (DEM) using standard terrain algorithms;
does not access any microclimate simulation internals.
"""

from typing import Dict, List
import numpy as np

from backend.sim.parcels import Parcel, ParcelNetwork
from backend.sim.terrain import SyntheticTerrain


class DEMFeatureExtractor:
    """Extracts standard geomorphometric covariates from the DEM."""

    def __init__(self, terrain: SyntheticTerrain, parcels: List[Parcel]):
        self.terrain = terrain
        self.parcels = parcels
        self.coords = np.array([[p.centroid_x, p.centroid_y] for p in parcels])
        self.elevs = np.array([p.elevation for p in parcels])
        self.rel_elevs = np.array([p.rel_elev for p in parcels])
        self.slopes = np.array([p.slope_deg for p in parcels])
        self._compute_tpi()
        self._compute_convergence()

    def _compute_tpi(self, radius_m: float = 600.0) -> None:
        """Topographic Position Index (TPI): elevation relative to local neighborhood mean."""
        dist_sq = np.sum((self.coords[:, None, :] - self.coords[None, :, :]) ** 2, axis=-1)
        neighbor_mask = (dist_sq <= radius_m ** 2) & (dist_sq > 0)
        
        # Mean neighbor elevation
        neighbor_sums = np.sum(neighbor_mask * self.elevs[None, :], axis=-1)
        neighbor_counts = np.maximum(1, np.sum(neighbor_mask, axis=-1))
        local_means = neighbor_sums / neighbor_counts
        
        self.tpi = self.elevs - local_means

    def _compute_convergence(self) -> None:
        """Standard terrain convergence index (Laplacian of elevation)."""
        # Approximated by divergence of slope gradient
        c_vals = []
        delta = 25.0
        for p in self.parcels:
            z_c = p.elevation
            z_n = self.terrain.elevation(p.centroid_x, p.centroid_y + delta)
            z_s = self.terrain.elevation(p.centroid_x, p.centroid_y - delta)
            z_e = self.terrain.elevation(p.centroid_x + delta, p.centroid_y)
            z_w = self.terrain.elevation(p.centroid_x - delta, p.centroid_y)
            # 2nd derivative / curvature proxy: positive indicates convex ridge, negative indicates concave trough/hollow
            laplacian = float((z_n + z_s + z_e + z_w - 4.0 * z_c) / (delta ** 2))
            c_vals.append(laplacian)
        self.convergence = np.array(c_vals)

    def get_parcel_feature_matrix(self, variant: str = "A") -> np.ndarray:
        """Return feature matrix for all parcels.
        
        Variant A: [x, y, rel_elev]
        Variant B: [x, y, rel_elev, slope, tpi, convergence]
        """
        if variant == "A":
            return np.column_stack([self.coords[:, 0], self.coords[:, 1], self.rel_elevs])
        else:
            return np.column_stack([
                self.coords[:, 0],
                self.coords[:, 1],
                self.rel_elevs,
                self.slopes,
                self.tpi,
                self.convergence,
            ])

    def get_point_features(self, x: float, y: float, variant: str = "A") -> np.ndarray:
        """Extract features for an arbitrary sensor coordinate."""
        feats = self.terrain.get_features(x, y)
        if variant == "A":
            return np.array([x, y, feats.rel_elev])
        
        # Compute TPI for point relative to parcel network
        dist_sq = (self.coords[:, 0] - x)**2 + (self.coords[:, 1] - y)**2
        mask = (dist_sq <= 600.0**2)
        local_mean = np.mean(self.elevs[mask]) if np.any(mask) else feats.elevation
        tpi_pt = feats.elevation - local_mean

        delta = 25.0
        z_c = feats.elevation
        z_n = self.terrain.elevation(x, y + delta)
        z_s = self.terrain.elevation(x, y - delta)
        z_e = self.terrain.elevation(x + delta, y)
        z_w = self.terrain.elevation(x - delta, y)
        laplacian = float((z_n + z_s + z_e + z_w - 4.0 * z_c) / (delta ** 2))

        return np.array([x, y, feats.rel_elev, feats.slope_deg, tpi_pt, laplacian])
