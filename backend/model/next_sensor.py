"""Next-best sensor placement engine using decision-focused uncertainty reduction.

Algorithm:
----------
Revised in Phase 3b: Rather than unweighted maximum variance (which tends to explore
extreme geometric boundary corners of the map), the decision-focused rule targets
uncertainty reduction specifically for parcels vulnerable to freezing:
  J(c) = sum_i w_i * max(0, sigma_i - sigma_i^new(c))
where w_i = E_training[exp(-0.5 * (mu_i / 1.8)^2)] is the frequency with which parcel i
lies near the freezing threshold (computed strictly from the model's predictions on
training nights 002-060, NEVER the hidden ground truth).
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.model.gp import UniversalKrigingGP
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork


class NextSensorOptimizer:
    """Finds optimal new weather station locations targeting frost decision boundaries."""

    def __init__(
        self,
        parcels_net: ParcelNetwork,
        decision_weights: Optional[np.ndarray] = None,
    ):
        self.parcels_net = parcels_net
        self.parcels = parcels_net.parcels
        self.parcel_coords = np.array([
            [p.centroid_x, p.centroid_y, p.rel_elev] for p in self.parcels
        ])
        
        # If decision weights are provided (precomputed from training nights), use them;
        # otherwise default to uniform
        if decision_weights is not None:
            self.weights = np.asarray(decision_weights, dtype=np.float64)
        else:
            # Fallback: weight parcels with lower relative elevation higher (natural frost susceptibility)
            rel_elevs = np.array([p.rel_elev for p in self.parcels])
            self.weights = np.exp(-0.5 * (rel_elevs / 35.0) ** 2)

    def find_next_sensor(
        self,
        current_model: UniversalKrigingGP,
        current_sensors_net: SensorNetwork,
        current_temps: np.ndarray,
    ) -> Dict[str, Any]:
        """Identify candidate parcel maximizing threshold-weighted variance reduction."""
        # 1. Current network-wide uncertainties
        _, current_sigmas = current_model.predict(self.parcel_coords)
        mean_sigma_current = float(np.mean(current_sigmas))

        # Filter out parcels that are within 250m of an existing station
        st_coords = np.array([[s.x, s.y] for s in current_sensors_net.stations])
        dists_to_stations = np.hypot(
            self.parcel_coords[:, None, 0] - st_coords[None, :, 0],
            self.parcel_coords[:, None, 1] - st_coords[None, :, 1],
        )
        min_dist_to_st = np.min(dists_to_stations, axis=-1)
        valid_candidate_mask = min_dist_to_st >= 250.0

        current_sensor_coords = np.array([
            [s.x, s.y, s.rel_elev] for s in current_sensors_net.stations
        ])

        # 2. Score each candidate by decision-weighted variance reduction
        scores = np.zeros(len(self.parcels), dtype=np.float64)
        sigmas_after_candidate = []

        for i in range(len(self.parcels)):
            if not valid_candidate_mask[i]:
                scores[i] = -1.0
                sigmas_after_candidate.append(current_sigmas)
                continue

            # Hypothetical addition
            hypothetical_coords = np.vstack([current_sensor_coords, self.parcel_coords[i]])
            dummy_y = np.append(current_temps, float(np.mean(current_temps)))

            hypothetical_model = UniversalKrigingGP()
            hypothetical_model.fit(hypothetical_coords, dummy_y)

            _, new_sigmas = hypothetical_model.predict(self.parcel_coords)
            sigmas_after_candidate.append(new_sigmas)

            # Weighted variance reduction across all parcels
            delta_sigma = np.maximum(0.0, current_sigmas - new_sigmas)
            scores[i] = float(np.sum(self.weights * delta_sigma))

        best_idx = int(np.argmax(scores))
        best_parcel = self.parcels[best_idx]
        best_new_sigmas = sigmas_after_candidate[best_idx]
        mean_sigma_after = float(np.mean(best_new_sigmas))
        sigma_reduction = max(0.0, float(mean_sigma_current - mean_sigma_after))

        # Top 5 ranked candidates
        sorted_indices = np.argsort(-scores)[:5]
        ranked = [
            {
                "parcel_id": self.parcels[i].id,
                "name": self.parcels[i].name,
                "lon": self.parcels[i].centroid_lon,
                "lat": self.parcels[i].centroid_lat,
                "rel_elev": self.parcels[i].rel_elev,
                "current_sigma": round(float(current_sigmas[i]), 2),
                "decision_score": round(float(scores[i]), 3),
            }
            for i in sorted_indices
        ]

        return {
            "best_parcel_id": best_parcel.id,
            "best_parcel_name": best_parcel.name,
            "best_coords": [best_parcel.centroid_lon, best_parcel.centroid_lat],
            "best_metric_coords": [best_parcel.centroid_x, best_parcel.centroid_y],
            "rel_elev": best_parcel.rel_elev,
            "elevation": best_parcel.elevation,
            "current_sigma": round(float(current_sigmas[best_idx]), 2),
            "expected_new_local_sigma": round(float(best_new_sigmas[best_idx]), 2),
            "network_mean_sigma_current": round(mean_sigma_current, 3),
            "network_mean_sigma_after": round(mean_sigma_after, 3),
            "sigma_reduction": round(sigma_reduction, 3),
            "decision_score": round(float(scores[best_idx]), 3),
            "ranked_candidates": ranked,
        }
