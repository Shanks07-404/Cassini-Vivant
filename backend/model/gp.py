"""Universal Kriging Gaussian Process with embedded linear lapse trend and frozen hyperparameters.

Key mathematical properties:
1. Linear elevation lapse trend is embedded INSIDE the Gaussian Process (Universal Kriging /
   semi-parametric GP with diffuse prior on trend coefficients beta).
   Posterior covariance includes trend parameter estimation uncertainty:
   cov(f_*) = K_** - K_*^T K_y^-1 K_* + R^T (H^T K_y^-1 H)^-1 R
   where R = H_* - H^T K_y^-1 K_*.
   As a result, predictive sigma grows naturally when extrapolating into elevation
   ranges unobserved by sensors (such as low-lying valley frost pockets).
2. Hyperparameters are learned ONCE from training nights 002-060 and strictly FROZEN.
   Because hyperparameters are fixed, conditioning on additional sensor stations is
   strictly positive semi-definite information: predictive variance is monotonically
   non-increasing as sensors are added.
3. night_001 is held-out and strictly excluded from hyperparameter learning.
"""

from typing import List, Optional, Tuple
import numpy as np


class UniversalKrigingGP:
    """Gaussian Process with explicit linear elevation trend (Universal Kriging).
    
    Default Pipeline: Farm-Only Calibration (Regime b - LOOCV on 6 farm stations):
      - l_xy = 527.2m (horizontal spatial correlation length)
      - l_h  = 148.9m (vertical topographic lapse scale)
      - sigma_f = 0.585°C (process standard deviation)
      - sigma_n = 0.101°C (sensor observation noise)
      - sigma_scale = 1.855 (calibrated empirical inflation factor)
    """

    def __init__(
        self,
        l_xy: float = 527.2,
        l_h: float = 148.9,
        sigma_f: float = 0.585,
        sigma_n: float = 0.101,
        sigma_scale: float = 1.855,
    ):
        self.l_xy = float(l_xy)
        self.l_h = float(l_h)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.sigma_scale = float(sigma_scale)

        # Fitted parameters
        self.X_train: Optional[np.ndarray] = None
        self.H_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.A_inv: Optional[np.ndarray] = None
        self.beta_hat: Optional[np.ndarray] = None
        self.res_train: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Anisotropic spatial RBF kernel over horizontal distance and relative elevation."""
        d_xy2 = (X1[:, 0, None] - X2[None, :, 0]) ** 2 + (X1[:, 1, None] - X2[None, :, 1]) ** 2
        d_h2 = (X1[:, 2, None] - X2[None, :, 2]) ** 2
        return (self.sigma_f ** 2) * np.exp(-0.5 * (d_xy2 / (self.l_xy ** 2) + d_h2 / (self.l_h ** 2)))

    def fit(self, sensor_coords: np.ndarray, sensor_temps: np.ndarray) -> "UniversalKrigingGP":
        """Condition the GP on sensor observations.
        
        sensor_coords: (N, 3) matrix of [x, y, rel_elev]
        sensor_temps: (N,) vector of temperatures
        """
        self.X_train = np.asarray(sensor_coords, dtype=np.float64)
        self.y_train = np.asarray(sensor_temps, dtype=np.float64)
        n = len(self.X_train)

        # Basis function matrix H = [1, rel_elev]
        self.H_train = np.column_stack([np.ones(n), self.X_train[:, 2]])

        # Covariance matrix with observation noise
        K_ff = self._kernel(self.X_train, self.X_train) + (self.sigma_n ** 2) * np.eye(n)
        self.K_inv = np.linalg.inv(K_ff)

        # Generalized Least Squares (GLS) estimate of trend coefficients beta
        A = self.H_train.T @ self.K_inv @ self.H_train
        self.A_inv = np.linalg.inv(A)
        self.beta_hat = self.A_inv @ (self.H_train.T @ self.K_inv @ self.y_train)

        # Residuals
        self.res_train = self.y_train - self.H_train @ self.beta_hat
        return self

    def predict(self, target_coords: np.ndarray, recalibrate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior mean mu and posterior standard deviation sigma at target points."""
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before predict()")

        X_test = np.asarray(target_coords, dtype=np.float64)
        m = len(X_test)
        H_test = np.column_stack([np.ones(m), X_test[:, 2]])

        K_fs = self._kernel(self.X_train, X_test)  # (N, M)
        K_ss_diag = np.full(m, self.sigma_f ** 2)

        # 1. Posterior Mean: trend + GP residual mean
        mu_trend = H_test @ self.beta_hat
        mu_res = K_fs.T @ self.K_inv @ self.res_train
        mu = mu_trend + mu_res

        # 2. Posterior Variance: GP variance + Trend parameter estimation variance
        v_gp = K_ss_diag - np.sum(K_fs * (self.K_inv @ K_fs), axis=0)
        # Parameter uncertainty term R^T (H^T K_inv H)^-1 R
        R = H_test.T - self.H_train.T @ self.K_inv @ K_fs  # (2, M)
        v_trend = np.sum(R * (self.A_inv @ R), axis=0)  # (M,)

        total_variance = np.maximum(v_gp + v_trend, self.sigma_n ** 2)
        scale = self.sigma_scale if recalibrate else 1.0
        sigma = np.sqrt(total_variance) * scale

        return np.round(mu, 2), np.round(sigma, 2)


class TerrainOnlyGP:
    """Baseline GP utilizing ONLY vertical relative elevation (no horizontal x/y coordinates)."""

    def __init__(self, l_h: float = 40.0, sigma_f: float = 0.85, sigma_n: float = 0.30):
        self.l_h = float(l_h)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.X_train: Optional[np.ndarray] = None
        self.H_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.A_inv: Optional[np.ndarray] = None
        self.beta_hat: Optional[np.ndarray] = None
        self.res_train: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        d_h2 = (X1[:, 2, None] - X2[None, :, 2]) ** 2
        return (self.sigma_f ** 2) * np.exp(-0.5 * (d_h2 / (self.l_h ** 2)))

    def fit(self, sensor_coords: np.ndarray, sensor_temps: np.ndarray) -> "TerrainOnlyGP":
        self.X_train = np.asarray(sensor_coords, dtype=np.float64)
        y = np.asarray(sensor_temps, dtype=np.float64)
        n = len(self.X_train)

        self.H_train = np.column_stack([np.ones(n), self.X_train[:, 2]])
        K_ff = self._kernel(self.X_train, self.X_train) + (self.sigma_n ** 2) * np.eye(n)
        self.K_inv = np.linalg.inv(K_ff)

        A = self.H_train.T @ self.K_inv @ self.H_train
        self.A_inv = np.linalg.inv(A)
        self.beta_hat = self.A_inv @ (self.H_train.T @ self.K_inv @ y)
        self.res_train = y - self.H_train @ self.beta_hat
        return self

    def predict(self, target_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_test = np.asarray(target_coords, dtype=np.float64)
        m = len(X_test)
        H_test = np.column_stack([np.ones(m), X_test[:, 2]])

        K_fs = self._kernel(self.X_train, X_test)
        K_ss_diag = np.full(m, self.sigma_f ** 2)

        mu_trend = H_test @ self.beta_hat
        mu_res = K_fs.T @ self.K_inv @ self.res_train
        mu = mu_trend + mu_res

        v_gp = K_ss_diag - np.sum(K_fs * (self.K_inv @ K_fs), axis=0)
        R = H_test.T - self.H_train.T @ self.K_inv @ K_fs
        v_trend = np.sum(R * (self.A_inv @ R), axis=0)

        total_variance = np.maximum(v_gp + v_trend, self.sigma_n ** 2)
        sigma = np.sqrt(total_variance)
        return np.round(mu, 2), np.round(sigma, 2)
