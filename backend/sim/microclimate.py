"""Deterministic physics-based microclimate truth generator for Cassini Vivant.

Physical Rationale:
-------------------
On clear, calm spring nights (radiation frost events):
1. Radiative cooling drives synoptic temperature drop across the region.
2. Cold, dense air drains down slopes into low-lying pockets (katabatic flow).
3. Inversion layers form: valley floor temperatures plummet 3-5°C below hilltops.
4. Tributary hollow drainage: convergent talwegs channel cold air down through
   specific side ravines, cooling certain mid-slope parcels below 0°C.
   (Note: this non-linear geometric drainage is unobserved by simple linear elevation
   models, ensuring the truth cannot be trivially reconstructed from rel_elev alone.)
5. Spatially correlated micro-variations (thermal inertia, soil moisture, hedgerows)
   are simulated via a 2D Gaussian Random Field (RBF kernel, length scale 750m).
6. Sensor readings add realistic Gaussian noise (sd ~0.25°C) and occasional dropouts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.sim.parcels import Parcel, ParcelNetwork
from backend.sim.sensors import SensorNetwork, Station
from backend.sim.terrain import SyntheticTerrain


@dataclass
class MicroclimateProfile:
    """Atmospheric conditions governing a specific night."""
    night_id: str
    is_hero_night: bool
    t_min_regional: float       # Ambient regional minimum temperature (°C)
    t_max_regional: float       # Evening starting temperature at 20:00 (°C)
    alpha_pool: float           # Katabatic cold pooling inversion strength (°C)
    h0_pool: float              # Vertical height scale of cold pool layer (m)
    alpha_drain: float          # Hollow convergence drainage cooling strength (°C)
    clear_calm_factor: float    # Radiative cooling efficiency factor [0.0 - 1.0]
    spatial_seed: int           # Deterministic seed for Gaussian Random Field


class MicroclimateSimulator:
    """Microclimate truth engine calculating hourly temperatures across parcels and sensors."""

    HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # 13 hourly frames

    def __init__(
        self,
        parcels: Optional[ParcelNetwork] = None,
        sensors: Optional[SensorNetwork] = None,
        terrain: Optional[SyntheticTerrain] = None,
    ):
        self.parcels_net = parcels or ParcelNetwork()
        self.sensors_net = sensors or SensorNetwork()
        self.terrain = terrain or SyntheticTerrain()
        self.parcels = self.parcels_net.parcels

        # Precompute spatial covariance matrix across parcel centroids
        self.parcel_coords = np.array([[p.centroid_x, p.centroid_y] for p in self.parcels])
        self._init_covariance()

    def _init_covariance(self) -> None:
        """Precompute Cholesky factor of spatial RBF kernel (length scale = 750m, sigma = 0.65°C)."""
        dist_sq = np.sum((self.parcel_coords[:, None, :] - self.parcel_coords[None, :, :]) ** 2, axis=-1)
        self.length_scale = 750.0
        self.sigma_spatial = 0.65
        self.K_parcels = (self.sigma_spatial ** 2) * np.exp(-dist_sq / (2.0 * self.length_scale ** 2))
        # Add jitter for numerical stability
        self.K_parcels += 1e-5 * np.eye(len(self.parcels))
        self.L_parcels = np.linalg.cholesky(self.K_parcels)
        self.K_inv = np.linalg.inv(self.K_parcels)

    def cooling_curve(self, hour_idx: int) -> float:
        """Progress of radiative cooling across the 13 time steps [0.0 at 20:00 to 1.0 at 04:00]."""
        # hour_idx 0 = 20:00, 8 = 04:00, 12 = 08:00
        if hour_idx <= 8:
            # Radiative cooling accelerates into the pre-dawn hours
            return float((hour_idx / 8.0) ** 0.85)
        else:
            # Sun rises, cold pool begins to dissipate
            post_dawn = (hour_idx - 8) / 4.0
            return float(1.0 - 0.55 * post_dawn)

    def regional_temp(self, profile: MicroclimateProfile, hour_idx: int) -> float:
        """Synoptic regional temperature for hour index."""
        c = self.cooling_curve(hour_idx)
        return float(profile.t_max_regional - c * (profile.t_max_regional - profile.t_min_regional))

    def hollow_drainage_metric(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Non-linear geometric index of convergent cold-air drainage along tributary hollows."""
        wave = np.sin(2.0 * np.pi * x / 1700.0) * np.cos(2.0 * np.pi * y / 1500.0)
        metric = np.maximum(0.0, -wave / 0.75) ** 1.3
        return metric

    def generate_random_field(self, seed: int) -> np.ndarray:
        """Deterministic 2D Gaussian Random Field realization for a night."""
        rng = np.random.RandomState(seed)
        z = rng.randn(len(self.parcels))
        return self.L_parcels @ z

    def simulate_night(
        self, profile: MicroclimateProfile
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Simulate microclimate truth across all 13 hours.
        
        Returns:
          - parcel_temps: (13, N_parcels) array of true temperatures
          - sensor_readings: (13, N_sensors) array of noisy station readings
          - regional_temps: (13,) array of regional ambient temperatures
          - frozen_parcels_set: list of parcel IDs whose true temp dropped below 0°C
        """
        n_hours = len(self.HOURS)
        n_parcels = len(self.parcels)
        stations = self.sensors_net.stations
        n_sensors = len(stations)

        grf_parcels = self.generate_random_field(profile.spatial_seed)

        # Interpolate GRF to station coordinates
        st_coords = np.array([[s.x, s.y] for s in stations])
        dist_st = np.sum((st_coords[:, None, :] - self.parcel_coords[None, :, :]) ** 2, axis=-1)
        K_cross = (self.sigma_spatial ** 2) * np.exp(-dist_st / (2.0 * self.length_scale ** 2))
        grf_sensors = K_cross @ (self.K_inv @ grf_parcels)

        parcel_temps = np.zeros((n_hours, n_parcels), dtype=np.float64)
        sensor_readings = np.zeros((n_hours, n_sensors), dtype=np.float64)
        regional_temps = np.zeros(n_hours, dtype=np.float64)

        # Precompute static spatial parcel features
        rel_elevs = np.array([p.rel_elev for p in self.parcels])
        slopes = np.array([p.slope_deg for p in self.parcels])
        hollows = self.hollow_drainage_metric(
            np.array([p.centroid_x for p in self.parcels]),
            np.array([p.centroid_y for p in self.parcels]),
        )

        sensor_noise_rng = np.random.RandomState(profile.spatial_seed + 999)

        for h_idx in range(n_hours):
            c = self.cooling_curve(h_idx) * profile.clear_calm_factor
            t_reg = self.regional_temp(profile, h_idx)
            regional_temps[h_idx] = t_reg

            # 1. Parcel Temperatures
            pool = - profile.alpha_pool * c * np.exp(-rel_elevs / profile.h0_pool) / (1.0 + 0.04 * slopes)
            drain = - profile.alpha_drain * c * hollows
            t_true_parcels = t_reg + pool + drain + c * grf_parcels
            parcel_temps[h_idx, :] = np.round(t_true_parcels, 2)

            # 2. Sensor Readings (Ground truth at sensor coords + measurement noise)
            for s_idx, st in enumerate(stations):
                feats = self.terrain.get_features(st.x, st.y)
                st_pool = - profile.alpha_pool * c * np.exp(-feats.rel_elev / profile.h0_pool) / (1.0 + 0.04 * feats.slope_deg)
                st_hollow = float(self.hollow_drainage_metric(st.x, st.y))
                st_drain = - profile.alpha_drain * c * st_hollow
                t_true_sensor = t_reg + st_pool + st_drain + c * grf_sensors[s_idx]
                
                # Sensor noise (sd = 0.30°C per specification)
                noise = sensor_noise_rng.normal(0.0, 0.30)
                sensor_readings[h_idx, s_idx] = round(float(t_true_sensor + noise), 2)

        # Determine ground-truth frozen parcel set (minimum temp across all hours < 0°C)
        min_temps_per_parcel = np.min(parcel_temps, axis=0)
        frozen_parcels_set = [self.parcels[i].id for i in range(n_parcels) if min_temps_per_parcel[i] < 0.0]

        return parcel_temps, sensor_readings, regional_temps, frozen_parcels_set
