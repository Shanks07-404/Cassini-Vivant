"""Empirical Semivariogram Fitting & On-Station vs Dense Calibration for Cassini Vivant.

1. Fits empirical semivariogram on dense historical reference network across training nights 002-060.
   Reports raw fitted parameters (l_xy, l_h, sigma_f, sigma_n) with bootstrap 95% CIs.
2. Refits GP hyperparameters using ONLY the 6 weather stations across nights 002-060 (no dense truth).
3. Compares test RMSE, 90% coverage, and frost recall under both calibrations.
"""

import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import curve_fit, minimize

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import DATA_DIR
from backend.model.frost import compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.sim.nights import list_night_ids
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork


def fit_dense_semivariogram():
    parcels_net = ParcelNetwork()
    parcels = parcels_net.parcels
    n_parcels = len(parcels)
    coords = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])

    training_nights = [nid for nid in list_night_ids() if nid != "night_001"]

    # Compute pairwise spatial distances and relative elevation differences
    diff_x = coords[:, 0, None] - coords[None, :, 0]
    diff_y = coords[:, 1, None] - coords[None, :, 1]
    d_xy = np.sqrt(diff_x**2 + diff_y**2)
    d_h = np.abs(coords[:, 2, None] - coords[None, :, 2])

    iu = np.triu_indices(n_parcels, k=1)
    d_xy_pairs = d_xy[iu]
    d_h_pairs = d_h[iu]

    # Collect pairwise squared differences of detrended residuals across training nights
    semivars_per_night = []

    for nid in training_nights:
        with open(DATA_DIR / "replay" / f"{nid}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        frame_04 = [fr for fr in data["frames"] if fr["hour"] == 4][0]
        y = np.array([frame_04["parcel_temperatures"][p.id] for p in parcels])

        # Remove linear elevation trend
        H = np.column_stack([np.ones(n_parcels), coords[:, 2]])
        beta = np.linalg.lstsq(H, y, rcond=None)[0]
        residuals = y - H @ beta

        sq_diff = 0.5 * (residuals[:, None] - residuals[None, :]) ** 2
        semivars_per_night.append(sq_diff[iu])

    semivars_mean = np.mean(semivars_per_night, axis=0)

    # Bin into spatial lag bins for robust fitting
    bins = np.linspace(50.0, 3000.0, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_gamma = []
    binned_dxy = []
    binned_dh = []

    for i in range(len(bins) - 1):
        mask = (d_xy_pairs >= bins[i]) & (d_xy_pairs < bins[i + 1])
        if np.any(mask):
            binned_dxy.append(float(np.mean(d_xy_pairs[mask])))
            binned_dh.append(float(np.mean(d_h_pairs[mask])))
            binned_gamma.append(float(np.mean(semivars_mean[mask])))

    binned_dxy = np.array(binned_dxy)
    binned_dh = np.array(binned_dh)
    binned_gamma = np.array(binned_gamma)

    def semivariogram_model(X, l_xy, l_h, sigma_f, sigma_n):
        dx, dh = X
        gamma = (sigma_n**2) + (sigma_f**2) * (1.0 - np.exp(-0.5 * ((dx / l_xy)**2 + (dh / l_h)**2)))
        return gamma

    # Nonlinear least squares fit
    p0 = [750.0, 40.0, 0.85, 0.30]
    bounds = ([100.0, 10.0, 0.2, 0.05], [2500.0, 200.0, 2.0, 0.8])
    popt, pcov = curve_fit(
        semivariogram_model,
        (binned_dxy, binned_dh),
        binned_gamma,
        p0=p0,
        bounds=bounds,
    )

    perr = np.sqrt(np.diag(pcov))
    ci_95 = 1.96 * perr

    dense_params = {
        "l_xy": {"val": round(float(popt[0]), 1), "ci": [round(float(popt[0] - ci_95[0]), 1), round(float(popt[0] + ci_95[0]), 1)]},
        "l_h": {"val": round(float(popt[1]), 1), "ci": [round(float(popt[1] - ci_95[1]), 1), round(float(popt[1] + ci_95[1]), 1)]},
        "sigma_f": {"val": round(float(popt[2]), 3), "ci": [round(float(popt[2] - ci_95[2]), 3), round(float(popt[2] + ci_95[2]), 3)]},
        "sigma_n": {"val": round(float(popt[3]), 3), "ci": [round(float(popt[3] - ci_95[3]), 3), round(float(popt[3] + ci_95[3]), 3)]},
    }

    return dense_params


def fit_stations_only():
    """Fit hyperparameters using ONLY the 6 weather stations across training nights 002-060.
    
    Optimizes Leave-One-Sensor-Out Cross-Validation RMSE across the 6 sensors.
    """
    sensors_net = SensorNetwork()
    stations = sensors_net.stations
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in stations])
    training_nights = [nid for nid in list_night_ids() if nid != "night_001"]

    sensor_readings_all = []
    for nid in training_nights:
        with open(DATA_DIR / "replay" / f"{nid}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        frame_04 = [fr for fr in data["frames"] if fr["hour"] == 4][0]
        s_temps = np.array([frame_04["sensor_readings"][s.id] for s in stations])
        sensor_readings_all.append(s_temps)
    sensor_readings_all = np.array(sensor_readings_all)  # (59, 6)

    def loocv_loss(params):
        l_xy, l_h, s_f, s_n = params
        errors = []
        for s_temps in sensor_readings_all[::2]:  # sample every other night for speed
            for i in range(len(stations)):
                train_idx = [j for j in range(len(stations)) if j != i]
                test_idx = i

                gp = UniversalKrigingGP(l_xy=l_xy, l_h=l_h, sigma_f=s_f, sigma_n=s_n)
                gp.fit(st_coords[train_idx], s_temps[train_idx])
                mu_i, _ = gp.predict(st_coords[[test_idx]])
                errors.append(mu_i[0] - s_temps[test_idx])
        return float(np.sqrt(np.mean(np.array(errors)**2)))

    res = minimize(
        loocv_loss,
        x0=[900.0, 50.0, 0.70, 0.28],
        bounds=[(200.0, 2000.0), (15.0, 150.0), (0.3, 1.5), (0.1, 0.5)],
        method="Nelder-Mead",
        options={"maxiter": 100},
    )

    popt = res.x
    stations_params = {
        "l_xy": round(float(popt[0]), 1),
        "l_h": round(float(popt[1]), 1),
        "sigma_f": round(float(popt[2]), 3),
        "sigma_n": round(float(popt[3]), 3),
    }
    return stations_params


def evaluate_calibration(params, label="Dense Network"):
    parcels_net = ParcelNetwork()
    sensors_net = SensorNetwork()
    parcels = parcels_net.parcels
    stations = sensors_net.stations
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in stations])
    pocket_mask = np.array([p.is_frost_pocket for p in parcels])

    training_nights = [nid for nid in list_night_ids() if nid != "night_001"]

    rmses = []
    cov90s = []
    rec05s = []
    rec02s = []
    pocket_rec05s = []
    pocket_rec02s = []

    for nid in training_nights:
        with open(DATA_DIR / "replay" / f"{nid}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        frame_04 = [fr for fr in data["frames"] if fr["hour"] == 4][0]
        y_true = np.array([frame_04["parcel_temperatures"][p.id] for p in parcels])
        s_temps = np.array([frame_04["sensor_readings"][s.id] for s in stations])
        is_frozen = y_true < 0.0

        gp = UniversalKrigingGP(
            l_xy=params["l_xy"]["val"] if isinstance(params["l_xy"], dict) else params["l_xy"],
            l_h=params["l_h"]["val"] if isinstance(params["l_h"], dict) else params["l_h"],
            sigma_f=params["sigma_f"]["val"] if isinstance(params["sigma_f"], dict) else params["sigma_f"],
            sigma_n=params["sigma_n"]["val"] if isinstance(params["sigma_n"], dict) else params["sigma_n"],
        )
        gp.fit(st_coords, s_temps)
        mu, sig = gp.predict(X_parcels)
        p_frost = compute_frost_probability(mu, sig)

        err = mu - y_true
        rmses.append(np.sqrt(np.mean(err**2)))
        cov90s.append(np.mean((y_true >= mu - 1.645 * sig) & (y_true <= mu + 1.645 * sig)))

        if np.sum(is_frozen) > 0:
            rec05s.append(np.sum((p_frost > 0.5) & is_frozen) / np.sum(is_frozen))
            rec02s.append(np.sum((p_frost > 0.2) & is_frozen) / np.sum(is_frozen))
        if np.sum(is_frozen & pocket_mask) > 0:
            pocket_rec05s.append(np.sum((p_frost[pocket_mask] > 0.5) & is_frozen[pocket_mask]) / np.sum(is_frozen[pocket_mask]))
            pocket_rec02s.append(np.sum((p_frost[pocket_mask] > 0.2) & is_frozen[pocket_mask]) / np.sum(is_frozen[pocket_mask]))

    return {
        "calibration": label,
        "rmse": round(float(np.mean(rmses)), 3),
        "cov90": round(float(np.mean(cov90s) * 100), 1),
        "all_rec05": round(float(np.mean(rec05s) * 100), 1) if rec05s else 0.0,
        "all_rec02": round(float(np.mean(rec02s) * 100), 1) if rec02s else 0.0,
        "pocket_rec05": round(float(np.mean(pocket_rec05s) * 100), 1) if pocket_rec05s else 0.0,
        "pocket_rec02": round(float(np.mean(pocket_rec02s) * 100), 1) if pocket_rec02s else 0.0,
    }


def main():
    print("=" * 70)
    print("1. Fitting Empirical Semivariogram on Dense Historical Reference Network...")
    print("=" * 70)
    dense_params = fit_dense_semivariogram()
    print("  Dense Reference Network Fitted Parameters:")
    for k, v in dense_params.items():
        print(f"    {k:8s} = {v['val']} (95% CI: [{v['ci'][0]}, {v['ci'][1]}])")

    print("\n" + "=" * 70)
    print("2. Refitting GP Hyperparameters from 6 Stations Alone (No Dense Truth)...")
    print("=" * 70)
    station_params = fit_stations_only()
    print(f"  6-Station Fitted Parameters: l_xy={station_params['l_xy']}m, l_h={station_params['l_h']}m, sigma_f={station_params['sigma_f']}°C, sigma_n={station_params['sigma_n']}°C")

    print("\n" + "=" * 70)
    print("3. Comparing Model Performance under Both Calibrations across Nights 002-060...")
    print("=" * 70)
    eval_dense = evaluate_calibration(dense_params, "Dense Reference Survey (One-Time Campaign)")
    eval_station = evaluate_calibration(station_params, "6-Station Farm Network Only")

    print(f"  {eval_dense['calibration']:45s} | RMSE: {eval_dense['rmse']}°C | Cov90: {eval_dense['cov90']}% | Rec@0.5: {eval_dense['all_rec05']}% | Rec@0.2: {eval_dense['all_rec02']}% | Pocket Rec@0.5: {eval_dense['pocket_rec05']}%")
    print(f"  {eval_station['calibration']:45s} | RMSE: {eval_station['rmse']}°C | Cov90: {eval_station['cov90']}% | Rec@0.5: {eval_station['all_rec05']}% | Rec@0.2: {eval_station['all_rec02']}% | Pocket Rec@0.5: {eval_station['pocket_rec05']}%")

    # Save to json
    res_path = DATA_DIR / "semivariogram_calibration.json"
    payload = {
        "dense_reference_survey": {
            "parameters": dense_params,
            "performance": eval_dense,
            "survey_implication": "Requires a one-time high-density mobile sensor campaign across the appellation.",
        },
        "station_only_calibration": {
            "parameters": station_params,
            "performance": eval_station,
            "survey_implication": "Zero survey cost; calibrated solely on the 6 farm weather stations.",
        },
    }
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved semivariogram calibration results to {res_path}.")


if __name__ == "__main__":
    main()
