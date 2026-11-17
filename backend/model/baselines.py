"""Baseline microclimate models conforming to uniform prediction interface."""

from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression

from backend.model.gp import TerrainOnlyGP


class NearestSensorBaseline:
    """Predict temperature from the spatially nearest active weather station."""

    def __init__(self):
        self.station_positions: Optional[np.ndarray] = None
        self.station_temps: Optional[np.ndarray] = None
        self.residual_std: float = 1.20

    def fit(self, positions: np.ndarray, temps: np.ndarray, **kwargs) -> "NearestSensorBaseline":
        self.station_positions = np.asarray(positions, dtype=np.float64)
        self.station_temps = np.asarray(temps, dtype=np.float64)
        if len(temps) > 1:
            self.residual_std = max(0.5, float(np.std(temps)))
        return self

    def predict(self, target_positions: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_positions, dtype=np.float64)
        dists = np.hypot(
            targets[:, None, 0] - self.station_positions[None, :, 0],
            targets[:, None, 1] - self.station_positions[None, :, 1],
        )
        nearest_idx = np.argmin(dists, axis=-1)
        nearest_dist = np.min(dists, axis=-1)

        mu = self.station_temps[nearest_idx]
        sigma = np.clip(self.residual_std * (0.8 + nearest_dist / 2000.0), 0.4, 2.5)
        return np.round(mu, 2), np.round(sigma, 2)


class IDWBaseline:
    """Inverse Distance Weighting (IDW) interpolation (power = 2.0)."""

    def __init__(self, power: float = 2.0):
        self.power = power
        self.station_positions: Optional[np.ndarray] = None
        self.station_temps: Optional[np.ndarray] = None
        self.residual_std: float = 1.0

    def fit(self, positions: np.ndarray, temps: np.ndarray, **kwargs) -> "IDWBaseline":
        self.station_positions = np.asarray(positions, dtype=np.float64)
        self.station_temps = np.asarray(temps, dtype=np.float64)
        if len(temps) > 1:
            self.residual_std = max(0.4, float(np.std(temps)))
        return self

    def predict(self, target_positions: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_positions, dtype=np.float64)
        dists = np.hypot(
            targets[:, None, 0] - self.station_positions[None, :, 0],
            targets[:, None, 1] - self.station_positions[None, :, 1],
        )
        eps = 1e-4
        weights = 1.0 / (np.maximum(dists, eps) ** self.power)
        weights_norm = weights / np.sum(weights, axis=-1, keepdims=True)

        mu = np.sum(weights_norm * self.station_temps[None, :], axis=-1)
        min_dist = np.min(dists, axis=-1)
        sigma = np.clip(self.residual_std * (0.6 + min_dist / 1800.0), 0.35, 2.2)
        return np.round(mu, 2), np.round(sigma, 2)


class TerrainLinearBaseline:
    """Ordinary Least Squares linear regression on relative elevation (rel_elev)."""

    def __init__(self):
        self.reg = LinearRegression()
        self.residual_std: float = 0.85

    def fit(
        self,
        positions: np.ndarray,
        temps: np.ndarray,
        sensor_features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TerrainLinearBaseline":
        if sensor_features is not None:
            X = sensor_features[:, [2]]  # rel_elev column
        else:
            X = positions[:, [0]]
        self.reg.fit(X, temps)
        preds = self.reg.predict(X)
        residuals = temps - preds
        if len(temps) > 2:
            self.residual_std = max(0.35, float(np.std(residuals)))
        return self

    def predict(
        self,
        target_positions: np.ndarray,
        target_features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if target_features is not None:
            X_t = target_features[:, [2]]
        else:
            X_t = target_positions[:, [0]]
        mu = self.reg.predict(X_t)
        sigma = np.full(len(mu), self.residual_std)
        return np.round(mu, 2), np.round(sigma, 2)


# Export TerrainOnlyGP under the baseline interface
__all__ = [
    "NearestSensorBaseline",
    "IDWBaseline",
    "TerrainLinearBaseline",
    "TerrainOnlyGP",
]
