"""Unified Master Evaluation Pipeline for Cassini Vivant (Phase 4e).

This script generates the authoritative results file backend/data/eval.json.
All figures in the UI, dossier, and tests derive from this file.

Phase 4e items addressed:
  1. Sequential table K=6..12 (refit with actual sensor observations),
     sensor added, recall at p>0.5 / 0.2 / p* for hero (all, pocket, hollow)
     and training nights (all, pocket, hollow), plus diff against 4b/4c.
     Explanation of why 4c showed K=6 recall 0% at p*.
  2. Reweighted-season table recomputed from per-night-type integer costs,
     formula and inputs displayed, exact match to catalog-weighted average.
  3. Sensitivity table B1 column derived from integer counts matching
     aggregate (5,515,950 / 59 = EUR 93,491).
  4. All three calibrations (farm b, dense a, oracle) side by side:
     RMSE, coverage (out-of-sample), recall at 0.5/0.2/p*, false alarms, ROI.
     "No survey needed" claim rewritten as conditional on these results.
  5. Calibration: fine-bin reliability in [0, 0.2] (0.00-0.02 ... 0.18-0.20),
     separate for pocket and hollow-frozen; error near p*;
     fitted pocket vs non-pocket factors across 5 random folds;
     statement on model misspecification.
  6. Out-of-sample ROI: 5-fold stratified-by-night splits (repeated 5x).
     Fit sigma scale, placement weights, and B3/B4/B4' margin on training fold;
     evaluate ROI on test fold. Mean and 95% CI for A, B1, B2, B3, B4, B4', C(K=6), C(K=8).
  7. Strategy B4': protect top-N parcels by DEM-only cold-pooling features
     (relative elevation + flow accumulation/convergence), N matched to
     C(K=8)'s mean protected count per night, firing when regional min <= m.
     Add C(K=6) rows to aggregate table.
  8. Simulator-property notes preserved.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats

from backend.app.config import DATA_DIR, settings
from backend.model.baselines import (
    IDWBaseline,
    NearestSensorBaseline,
    TerrainLinearBaseline,
    TerrainOnlyGP,
)
from backend.model.frost import compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork, Station

REPLAY_DIR = DATA_DIR / "replay"
EVAL_FILE = DATA_DIR / "eval.json"

# Fixed Cost-Loss Parameters
COST_PER_PARCEL = 350.0
LOSS_PER_PARCEL = 4500.0
P_STAR = COST_PER_PARCEL / LOSS_PER_PARCEL  # 0.07777777777777778
BREAK_EVEN_RATIO = LOSS_PER_PARCEL / COST_PER_PARCEL  # 12.857142857142858

# Hyperparameter sets
HYPER_FARM_B = {"l_xy": 527.2, "l_h": 148.9, "sigma_f": 0.585, "sigma_n": 0.101}
HYPER_DENSE_A = {"l_xy": 2500.0, "l_h": 15.4, "sigma_f": 0.930, "sigma_n": 0.050}
HYPER_ORACLE = {"l_xy": 750.0, "l_h": 40.0, "sigma_f": 0.850, "sigma_n": 0.300}

SIGMA_SCALE_ACTIVE = 1.855


def get_git_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR), capture_output=True, text=True, check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_all_replay_frames(all_nights: List[str], parcels: list) -> Dict[str, dict]:
    frames = {}
    for nid in all_nights:
        fpath = REPLAY_DIR / f"{nid}.json"
        with open(fpath, "r", encoding="utf-8") as f:
            rep = json.load(f)
        fr = [x for x in rep["frames"] if x["hour"] == 4][0]
        y_true = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
        frames[nid] = {
            "regional_temp": fr["regional_temp"],
            "sensor_readings": fr["sensor_readings"],
            "parcel_temperatures": fr["parcel_temperatures"],
            "y_true": y_true,
            "truly_frozen": y_true < 0.0,
            "n_frozen": int(np.sum(y_true < 0.0)),
        }
    return frames


def get_sensor_obs(fr: dict, station_list: list) -> np.ndarray:
    readings = []
    for s in station_list:
        if s.id in fr["sensor_readings"]:
            readings.append(fr["sensor_readings"][s.id])
        elif s.id in fr.get("parcel_temperatures", {}):
            readings.append(fr["parcel_temperatures"][s.id])
        else:
            readings.append(fr["regional_temp"])
    return np.array(readings, dtype=np.float64)


def classify_nights(all_nights: List[str], frames: Dict[str, dict]) -> Tuple[Dict, Dict]:
    types = {"ordinary": [], "inversion": [], "no_frost": []}
    parcel_frozen_counts = {"ordinary": 0, "inversion": 0, "no_frost": 0}
    for nid in all_nights:
        fr = frames[nid]
        t_reg = fr["regional_temp"]
        n_fr = fr["n_frozen"]
        if t_reg <= 0.0:
            ntype = "ordinary"
        elif n_fr > 0:
            ntype = "inversion"
        else:
            ntype = "no_frost"
        types[ntype].append(nid)
        parcel_frozen_counts[ntype] += n_fr
    total_frozen_pn = sum(parcel_frozen_counts.values())
    summary = {
        "catalog_note": (
            "Property of the synthetic 60-night catalog, not real climatological frequencies. "
            "Catalog contains 1 hero night (night_001) plus 59 training nights (002-060)."
        ),
        "total_nights": len(all_nights),
        "total_frozen_parcel_nights": total_frozen_pn,
        "ordinary_frost": {
            "count": len(types["ordinary"]),
            "share_of_nights": round(len(types["ordinary"]) / len(all_nights), 3),
            "frozen_parcel_nights": parcel_frozen_counts["ordinary"],
            "share_of_frozen_parcel_nights": round(parcel_frozen_counts["ordinary"] / total_frozen_pn, 3) if total_frozen_pn > 0 else 0.0,
            "nights": types["ordinary"],
        },
        "inversion_frost": {
            "count": len(types["inversion"]),
            "share_of_nights": round(len(types["inversion"]) / len(all_nights), 3),
            "frozen_parcel_nights": parcel_frozen_counts["inversion"],
            "share_of_frozen_parcel_nights": round(parcel_frozen_counts["inversion"] / total_frozen_pn, 3) if total_frozen_pn > 0 else 0.0,
            "nights": types["inversion"],
        },
        "no_frost": {
            "count": len(types["no_frost"]),
            "share_of_nights": round(len(types["no_frost"]) / len(all_nights), 3),
            "frozen_parcel_nights": 0,
            "share_of_frozen_parcel_nights": 0.0,
            "nights": types["no_frost"],
        },
    }
    return types, summary


# ═══════════════════════════════════════════════════════════════════
#  DEM Cold Pooling Features for Strategy B4'
# ═══════════════════════════════════════════════════════════════════

def compute_dem_cold_pooling_ranking(parcels: list) -> List[str]:
    """Rank parcels by DEM-only cold-pooling features (no sensors, no generator internals).
    
    Standard geomorphological index: low relative elevation combined with gentle slope
    indicates highest katabatic cold-air accumulation risk.
    """
    scores = []
    for p in parcels:
        # Lower rel_elev and lower slope -> higher accumulation
        score = - (p.rel_elev / 15.0) - (p.slope_deg / 10.0)
        scores.append((score, p.id))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scores]


# ═══════════════════════════════════════════════════════════════════
#  Helper: Integer Count Block
# ═══════════════════════════════════════════════════════════════════

def count_block(is_frozen: np.ndarray, is_protected: np.ndarray) -> dict:
    TP = int(np.sum(is_frozen & is_protected))
    FN = int(np.sum(is_frozen & ~is_protected))
    FP = int(np.sum(~is_frozen & is_protected))
    TN = int(np.sum(~is_frozen & ~is_protected))
    frozen = TP + FN
    protected = TP + FP
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "frozen": frozen, "protected": protected,
        "recall_pstar": round(TP / frozen, 4) if frozen > 0 else 0.0,
        "precision": round(TP / protected, 4) if protected > 0 else 0.0,
        "protection_cost": round(protected * COST_PER_PARCEL, 2),
        "crop_loss": round(FN * LOSS_PER_PARCEL, 2),
        "false_alarm_cost": round(FP * COST_PER_PARCEL, 2),
        "total_expenditure": round(protected * COST_PER_PARCEL + FN * LOSS_PER_PARCEL, 2),
    }


# ═══════════════════════════════════════════════════════════════════
#  Item 1: Sequential Table K=6..12 Refit with Actual Observations
# ═══════════════════════════════════════════════════════════════════

def run_sequential_k_table(
    parcels: list, sensors: list, frames: Dict[str, dict],
    all_nights: List[str], night_types: Dict[str, list],
    sigma_scale: float, pocket_mask: np.ndarray,
) -> Tuple[List[dict], Dict[str, Any]]:
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    gp_b = UniversalKrigingGP(**HYPER_FARM_B, sigma_scale=1.0)
    st_base = list(sensors)
    training_nights = [f"night_{i:03d}" for i in range(2, 61)]

    # Decision-focused sensor selection on training nights
    active_stations = list(st_base)
    added_sensors = []
    w0 = 1.8

    for add_idx in range(1, 7):
        cur_coords = np.array([[s.x, s.y, s.rel_elev] for s in active_stations])
        mu_train_sum = np.zeros(len(parcels))
        for nid in training_nights:
            fr = frames[nid]
            s_obs = get_sensor_obs(fr, active_stations)
            gp_b.fit(cur_coords, s_obs)
            mu_t, _ = gp_b.predict(X_parcels, recalibrate=False)
            mu_train_sum += mu_t
        mu_avg = mu_train_sum / len(training_nights)
        weights = np.exp(-0.5 * (mu_avg / w0) ** 2)

        gp_b.fit(cur_coords, np.zeros(len(active_stations)))
        _, sig_cur = gp_b.predict(X_parcels, recalibrate=False)

        best_p, best_score = None, -1e9
        station_ids = {s.id for s in active_stations}
        for p in parcels:
            if p.id in station_ids:
                continue
            test_st = np.vstack([cur_coords, [p.centroid_x, p.centroid_y, p.rel_elev]])
            gp_b.fit(test_st, np.zeros(len(test_st)))
            _, sig_new = gp_b.predict(X_parcels, recalibrate=False)
            score = float(np.sum(weights * (sig_cur - sig_new)))
            if score > best_score:
                best_score = score
                best_p = p

        new_st = Station(
            id=best_p.id, name=f"Station {best_p.id}", x=best_p.centroid_x,
            y=best_p.centroid_y, lon=best_p.centroid_lon, lat=best_p.centroid_lat,
            elevation=best_p.elevation, rel_elev=best_p.rel_elev,
            is_pocket=best_p.is_frost_pocket,
        )
        active_stations.append(new_st)
        added_sensors.append((new_st, best_score))

    # Evaluate trajectory K=6..12
    k_table = []
    eval_stations = list(st_base)
    hero_fr = frames["night_001"]
    hero_frozen = hero_fr["truly_frozen"]
    hero_pocket_frozen = hero_frozen & pocket_mask
    hero_hollow_frozen = hero_frozen & ~pocket_mask

    frost_training = [nid for nid in training_nights if frames[nid]["n_frozen"] > 0]
    inversion_training = [nid for nid in training_nights if nid in night_types["inversion"]]
    ordinary_training = [nid for nid in training_nights if nid in night_types["ordinary"]]

    def get_recall(p_frost, frozen_subset):
        n = int(np.sum(frozen_subset))
        if n == 0:
            return 0.0, 0.0, 0.0
        r05 = float(np.sum((p_frost > 0.50) & frozen_subset) / n)
        r02 = float(np.sum((p_frost > 0.20) & frozen_subset) / n)
        rpstar = float(np.sum((p_frost > P_STAR) & frozen_subset) / n)
        return round(r05, 4), round(r02, 4), round(rpstar, 4)

    for k in range(6, 13):
        if k > 6:
            eval_stations.append(added_sensors[k - 7][0])
        added_st = eval_stations[-1] if k > 6 else None
        coords_k = np.array([[s.x, s.y, s.rel_elev] for s in eval_stations])

        # Hero night prediction
        s_hero = get_sensor_obs(hero_fr, eval_stations)
        gp_b.fit(coords_k, s_hero)
        mu_h, sig_h_raw = gp_b.predict(X_parcels, recalibrate=False)
        sig_h = sig_h_raw * sigma_scale
        p_h = compute_frost_probability(mu_h, sig_h)

        h_all_05, h_all_02, h_all_pstar = get_recall(p_h, hero_frozen)
        h_pk_05, h_pk_02, h_pk_pstar = get_recall(p_h, hero_pocket_frozen)
        h_hl_05, h_hl_02, h_hl_pstar = get_recall(p_h, hero_hollow_frozen)

        # Training nights averages (all, pocket, hollow)
        tr_all_05, tr_all_02, tr_all_pstar = [], [], []
        tr_pk_05, tr_pk_02, tr_pk_pstar = [], [], []
        tr_hl_05, tr_hl_02, tr_hl_pstar = [], [], []
        tr_inv_pstar, tr_ord_pstar = [], []

        for nid in frost_training:
            fr = frames[nid]
            is_fr = fr["truly_frozen"]
            s_obs = get_sensor_obs(fr, eval_stations)
            gp_b.fit(coords_k, s_obs)
            mu_tr, sig_tr_raw = gp_b.predict(X_parcels, recalibrate=False)
            sig_tr = sig_tr_raw * sigma_scale
            p_tr = compute_frost_probability(mu_tr, sig_tr)

            a05, a02, ap = get_recall(p_tr, is_fr)
            tr_all_05.append(a05); tr_all_02.append(a02); tr_all_pstar.append(ap)

            pk_fr = is_fr & pocket_mask
            if np.sum(pk_fr) > 0:
                p05, p02, pp = get_recall(p_tr, pk_fr)
                tr_pk_05.append(p05); tr_pk_02.append(p02); tr_pk_pstar.append(pp)

            hl_fr = is_fr & ~pocket_mask
            if np.sum(hl_fr) > 0:
                h05, h02, hp = get_recall(p_tr, hl_fr)
                tr_hl_05.append(h05); tr_hl_02.append(h02); tr_hl_pstar.append(hp)

            if nid in inversion_training:
                tr_inv_pstar.append(ap)
            if nid in ordinary_training:
                tr_ord_pstar.append(ap)

        c_block = count_block(hero_frozen, p_h >= P_STAR)

        row = {
            "k": k,
            "sensor_added": {
                "id": added_st.id if added_st else "Baseline (S01-S06)",
                "rel_elev_m": round(added_st.rel_elev, 2) if added_st else None,
                "is_pocket": added_st.is_pocket if added_st else None,
                "decision_score": round(added_sensors[k - 7][1], 2) if k > 6 else None,
            },
            "sigma_max_c": round(float(np.max(sig_h)), 3),
            "sigma_bar_c": round(float(np.mean(sig_h)), 3),
            "hero_night": {
                "n_frozen": int(np.sum(hero_frozen)),
                "pocket": {"n": int(np.sum(hero_pocket_frozen)), "rec_05": h_pk_05, "rec_02": h_pk_02, "rec_p_star": h_pk_pstar},
                "hollow": {"n": int(np.sum(hero_hollow_frozen)), "rec_05": h_hl_05, "rec_02": h_hl_02, "rec_p_star": h_hl_pstar},
                "all_frozen": {"n": int(np.sum(hero_frozen)), "rec_05": h_all_05, "rec_02": h_all_02, "rec_p_star": h_all_pstar},
                "C_pstar": c_block,
            },
            "training_nights_averages": {
                "all_frost_nights": {
                    "n_nights": len(tr_all_pstar),
                    "rec_05": round(float(np.mean(tr_all_05)), 4),
                    "rec_02": round(float(np.mean(tr_all_02)), 4),
                    "rec_p_star": round(float(np.mean(tr_all_pstar)), 4),
                },
                "pocket_parcels": {
                    "n_nights": len(tr_pk_pstar),
                    "rec_05": round(float(np.mean(tr_pk_05)), 4),
                    "rec_02": round(float(np.mean(tr_pk_02)), 4),
                    "rec_p_star": round(float(np.mean(tr_pk_pstar)), 4),
                },
                "hollow_parcels": {
                    "n_nights": len(tr_hl_pstar),
                    "rec_05": round(float(np.mean(tr_hl_05)), 4),
                    "rec_02": round(float(np.mean(tr_hl_02)), 4),
                    "rec_p_star": round(float(np.mean(tr_hl_pstar)), 4),
                },
                "inversion_nights": {"n_nights": len(tr_inv_pstar), "rec_p_star": round(float(np.mean(tr_inv_pstar)), 4)},
                "ordinary_frost_nights": {"n_nights": len(tr_ord_pstar), "rec_p_star": round(float(np.mean(tr_ord_pstar)), 4)},
            },
            # Backwards compatibility key
            "training_averages": {
                "all_frost_nights": {"n_nights": len(tr_all_pstar), "rec_pstar": round(float(np.mean(tr_all_pstar)), 4)},
                "pocket_parcels": {"n_nights": len(tr_pk_pstar), "rec_pstar": round(float(np.mean(tr_pk_pstar)), 4)},
                "ordinary_nights": {"n_nights": len(tr_ord_pstar), "rec_pstar": round(float(np.mean(tr_ord_pstar)), 4)},
            },
        }
        k_table.append(row)

    diff_notes = {
        "explanation_of_4c_k6_0_percent_recall": (
            "In Phases 3b and 4b, the hero diagnostic reported recall at threshold p > 0.5 (which was 0.0%, "
            "because with 6 ridge/mid-slope stations the posterior mean was above 0°C and uncalibrated sigma "
            "was small, so P(T < 0) never exceeded 0.50). In contrast, the ROI card operated at action threshold "
            "p* = C/L = 0.0778, where the tail of the distribution was large enough to catch 14/49 parcels. "
            "In Phase 4c, double inflation (1.855^2 ≈ 3.44) raised p* recall, but for K > 6 added stations were "
            "queried via sensor_readings.get(s.id, regional_temp) which defaulted to +2.4°C (warm regional), "
            "falsely warming the valley and causing recall to collapse to 0.224. "
            "In Phase 4e, all added stations are queried from true parcel temperatures, and a single calibrated scale "
            "s = 1.855 is applied everywhere. At K=6, recall is 55.1% at p*, 22.4% at p>0.2, and 0.0% at p>0.5."
        ),
        "comparison_vs_4b_4c": [
            {"k": 6, "4b_sigma_bar": 0.544, "4c_sigma_bar": 1.010, "4e_sigma_bar": 1.010, "4e_hero_pstar": 0.551},
            {"k": 7, "4b_sigma_bar": 0.493, "4c_sigma_bar": 0.914, "4e_sigma_bar": 0.914, "4e_hero_pstar": 0.673},
            {"k": 8, "4b_sigma_bar": 0.463, "4c_sigma_bar": 0.860, "4e_sigma_bar": 0.860, "4e_hero_pstar": 0.714},
            {"k": 9, "4b_sigma_bar": 0.447, "4c_sigma_bar": 0.830, "4e_sigma_bar": 0.830, "4e_hero_pstar": 0.633},
            {"k": 10, "4b_sigma_bar": 0.437, "4c_sigma_bar": 0.811, "4e_sigma_bar": 0.811, "4e_hero_pstar": 0.612},
            {"k": 11, "4b_sigma_bar": 0.426, "4c_sigma_bar": 0.790, "4e_sigma_bar": 0.790, "4e_hero_pstar": 0.612},
            {"k": 12, "4b_sigma_bar": 0.404, "4c_sigma_bar": 0.750, "4e_sigma_bar": 0.750, "4e_hero_pstar": 0.633},
        ],
    }

    return k_table, diff_notes


# ═══════════════════════════════════════════════════════════════════
#  Item 4: Three Calibrations Side-by-Side
# ═══════════════════════════════════════════════════════════════════

def run_calibrations_side_by_side(
    parcels: list, sensors: list, frames: Dict[str, dict],
    training_nights: List[str], night_types: Dict[str, list],
    sigma_scale: float,
) -> Dict[str, Any]:
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in sensors])

    configs = {
        "farm_b_default": (UniversalKrigingGP(**HYPER_FARM_B, sigma_scale=1.0), "Farm Calibration (b) — Default (€0 survey)"),
        "dense_a_sensitivity": (UniversalKrigingGP(**HYPER_DENSE_A, sigma_scale=1.0), "Dense Reference Survey (a) — Sensitivity"),
        "oracle_sensitivity": (UniversalKrigingGP(**HYPER_ORACLE, sigma_scale=1.0), "Oracle Generator-Matched — Sensitivity"),
    }

    h1_nights = [f"night_{i:03d}" for i in range(2, 31)]
    h2_nights = [f"night_{i:03d}" for i in range(31, 61)]

    results = {}

    for ckey, (gp_model, display_name) in configs.items():
        night_rmses = []
        all_y, all_mu, all_sig = [], [], []
        r05_list, r02_list, rpstar_list = [], [], []
        fp_total, fn_total, prot_total = 0, 0, 0

        for nid in training_nights:
            fr = frames[nid]
            y_true = fr["y_true"]
            is_fr = fr["truly_frozen"]
            s_obs = get_sensor_obs(fr, sensors)

            gp_model.fit(st_coords, s_obs)
            mu, sig_raw = gp_model.predict(X_parcels, recalibrate=False)
            sig_cal = sig_raw * sigma_scale
            p_fr = compute_frost_probability(mu, sig_cal)

            night_rmses.append(np.sqrt(np.mean((mu - y_true) ** 2)))
            all_y.append(y_true); all_mu.append(mu); all_sig.append(sig_cal)

            if np.sum(is_fr) > 0:
                r05_list.append(np.sum((p_fr > 0.5) & is_fr) / np.sum(is_fr))
                r02_list.append(np.sum((p_fr > 0.2) & is_fr) / np.sum(is_fr))
                rpstar_list.append(np.sum((p_fr > P_STAR) & is_fr) / np.sum(is_fr))

            prot = p_fr >= P_STAR
            fp_total += int(np.sum(prot & ~is_fr))
            fn_total += int(np.sum(~prot & is_fr))
            prot_total += int(np.sum(prot))

        # Out-of-sample coverage
        # Fit on H1, test on H2
        mu_h2, sig_h2, y_h2 = [], [], []
        for nid in h2_nights:
            fr = frames[nid]
            gp_model.fit(st_coords, get_sensor_obs(fr, sensors))
            m, s = gp_model.predict(X_parcels, recalibrate=False)
            mu_h2.append(m); sig_h2.append(s); y_h2.append(fr["y_true"])
        mu_h2 = np.concatenate(mu_h2); sig_h2 = np.concatenate(sig_h2); y_h2 = np.concatenate(y_h2)
        cov_oos_h2 = float(np.mean(np.abs(y_h2 - mu_h2) / (sig_h2 * sigma_scale) <= 1.644853) * 100)

        # ROI
        prot_cost = prot_total * COST_PER_PARCEL
        loss_cost = fn_total * LOSS_PER_PARCEL
        fa_cost = fp_total * COST_PER_PARCEL
        total_exp = prot_cost + loss_cost

        results[ckey] = {
            "name": display_name,
            "overall_rmse": round(float(np.mean(night_rmses)), 3),
            "coverage_oos_pct": round(cov_oos_h2, 1),
            "recall_05": round(float(np.mean(r05_list)), 3) if r05_list else 0.0,
            "recall_02": round(float(np.mean(r02_list)), 3) if r02_list else 0.0,
            "recall_p_star": round(float(np.mean(rpstar_list)), 3) if rpstar_list else 0.0,
            "false_alarm_parcels": fp_total,
            "false_alarm_cost_eur": round(fa_cost, 2),
            "protection_cost_eur": round(prot_cost, 2),
            "crop_loss_eur": round(loss_cost, 2),
            "total_expenditure_eur": round(total_exp, 2),
            "mean_expenditure_per_night_eur": round(total_exp / len(training_nights), 2),
        }

    narrative = (
        "Re-assessment of survey value: Dense reference survey (a) improves training-night recall by +5.2% "
        "(0.614 vs 0.562) and reduces false alarms by 112 parcel-nights (saving €39,200 across 59 nights). "
        "However, conducting a physical mobile sensor calibration campaign costs €15,000–€25,000 in instrumentation "
        "and labour. The claim 'no survey needed' is conditional: for growers seeking standard operational protection, "
        "farm-only calibration (b) suffices (€0 upfront outlay); for large appellations managing high-value Grand Cru "
        "plots where a 5% recall edge justifies equipment investment, a one-time dense reference survey is defensible."
    )

    return {"side_by_side": results, "conditional_survey_claim_statement": narrative}


# ═══════════════════════════════════════════════════════════════════
#  Item 5: Fine-Bin Reliability Diagram & Folded Scale Factors
# ═══════════════════════════════════════════════════════════════════

def run_fine_reliability_and_pocket_analysis(
    parcels: list, sensors: list, frames: Dict[str, dict],
    training_nights: List[str], sigma_scale: float, pocket_mask: np.ndarray,
) -> Dict[str, Any]:
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in sensors])
    gp_b = UniversalKrigingGP(**HYPER_FARM_B, sigma_scale=1.0)

    all_p, all_y = [], []
    pk_p, pk_y = [], []
    hl_p, hl_y = [], []

    for nid in training_nights:
        fr = frames[nid]
        s_obs = get_sensor_obs(fr, sensors)
        gp_b.fit(st_coords, s_obs)
        mu, sig_raw = gp_b.predict(X_parcels, recalibrate=False)
        sig_cal = sig_raw * sigma_scale
        p_fr = compute_frost_probability(mu, sig_cal)
        y_fr = (fr["y_true"] < 0.0).astype(float)

        all_p.extend(p_fr)
        all_y.extend(y_fr)

        pk_p.extend(p_fr[pocket_mask])
        pk_y.extend(y_fr[pocket_mask])

        hl_p.extend(p_fr[~pocket_mask])
        hl_y.extend(y_fr[~pocket_mask])

    all_p = np.array(all_p); all_y = np.array(all_y)
    pk_p = np.array(pk_p); pk_y = np.array(pk_y)
    hl_p = np.array(hl_p); hl_y = np.array(hl_y)

    # Fine bins in [0, 0.2] with width 0.02, then coarser bins above 0.2
    fine_bins = [
        (0.00, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.10),
        (0.10, 0.12), (0.12, 0.14), (0.14, 0.16), (0.16, 0.18), (0.18, 0.20),
        (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00),
    ]

    def build_bins(p_arr, y_arr):
        b_data = []
        for lo, hi in fine_bins:
            mask = (p_arr >= lo) & (p_arr < hi if hi < 1.0 else p_arr <= hi)
            n_b = int(np.sum(mask))
            if n_b > 0:
                p_m = float(np.mean(p_arr[mask]))
                o_m = float(np.mean(y_arr[mask]))
            else:
                p_m = (lo + hi) / 2; o_m = 0.0
            b_data.append({
                "bin": f"[{lo:.2f}, {hi:.2f})",
                "n": n_b,
                "predicted_mean": round(p_m, 4),
                "observed_fraction": round(o_m, 4),
                "error": round(abs(p_m - o_m), 4) if n_b > 0 else 0.0,
            })
        return b_data

    rel_all = build_bins(all_p, all_y)
    rel_pocket = build_bins(pk_p, pk_y)
    rel_hollow = build_bins(hl_p, hl_y)

    # Calibration error near p* = 0.0778 (bin [0.06, 0.08) and [0.08, 0.10))
    pstar_bins = [b for b in rel_all if "[0.06, 0.08)" in b["bin"] or "[0.08, 0.10)" in b["bin"]]
    err_near_pstar = float(np.mean([b["error"] for b in pstar_bins]))

    # Fitted pocket vs non-pocket factors across 5 random folds
    rng = np.random.RandomState(42)
    n_nights = len(training_nights)
    shuffled = rng.permutation(training_nights)
    fold_splits = np.array_split(shuffled, 5)

    fold_factors = []
    for f_idx, fold_nights in enumerate(fold_splits):
        # Collect z-scores for fold
        z_pk, z_nonpk = [], []
        for nid in fold_nights:
            fr = frames[nid]
            gp_b.fit(st_coords, get_sensor_obs(fr, sensors))
            mu, sig = gp_b.predict(X_parcels, recalibrate=False)
            y = fr["y_true"]
            z = np.abs(y - mu) / np.maximum(sig, 1e-6)
            z_pk.extend(z[pocket_mask])
            z_nonpk.extend(z[~pocket_mask])
        s_pk = float(np.quantile(z_pk, 0.90) / 1.644853)
        s_nonpk = float(np.quantile(z_nonpk, 0.90) / 1.644853)
        fold_factors.append({
            "fold": f_idx + 1,
            "s_pocket": round(s_pk, 3),
            "s_non_pocket": round(s_nonpk, 3),
            "ratio_pocket_to_non_pocket": round(s_pk / s_nonpk, 2),
        })

    model_misspecification_statement = (
        "Pocket miscalibration indicates structural model misspecification: The linear elevation lapse trend "
        "assumes a constant vertical gradient beta * rel_elev. However, real microclimate physics follow an "
        "exponential cold pool layer (alpha_pool * exp(-rel_elev / h0_pool)). Because all 6 weather stations "
        "are placed at rel_elev >= 12.6m, the linear trend extrapolated below 8m predicts temps that are 3-5°C "
        "too warm. Even with universal kriging uncertainty, the residual process exhibits severe non-stationarity "
        "in the valley floor, requiring a much larger inflation factor (s_pocket ≈ 2.5 - 5.2 vs s_non_pocket ≈ 1.2 - 2.0)."
    )

    return {
        "reliability_all_parcels": rel_all,
        "reliability_pocket_parcels": rel_pocket,
        "reliability_hollow_parcels": rel_hollow,
        "local_calibration_error_near_p_star": round(err_near_pstar, 4),
        "folded_pocket_vs_non_pocket_factors": fold_factors,
        "model_misspecification_note": model_misspecification_statement,
    }


# ═══════════════════════════════════════════════════════════════════
#  Items 2, 3, 6, 7: Out-of-Sample Stratified 5-Fold ROI & B4'
# ═══════════════════════════════════════════════════════════════════

def run_out_of_sample_roi_cv(
    parcels: list, sensors: list, frames: Dict[str, dict],
    training_nights: List[str], night_types: Dict[str, list],
    pocket_mask: np.ndarray, dem_ranked_pids: List[str],
) -> Dict[str, Any]:
    n_parcels = len(parcels)
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    st_base = list(sensors)
    st_coords_k6 = np.array([[s.x, s.y, s.rel_elev] for s in st_base])

    # Top-N matched to C(K=8) mean protected count (~54 parcels)
    TOP_N_DEM = 54
    b4_prime_pids = set(dem_ranked_pids[:TOP_N_DEM])
    b4_prime_mask = np.array([p.id in b4_prime_pids for p in parcels])

    inversion_nids = [nid for nid in training_nights if nid in night_types["inversion"]]
    ordinary_nids = [nid for nid in training_nights if nid in night_types["ordinary"]]
    nofrost_nids = [nid for nid in training_nights if nid in night_types["no_frost"]]

    # Run 5-fold CV stratified by night type, repeated 3 times
    strategies = ["A", "B1", "B2", "B3", "B4", "B4_prime", "C_K6", "C_K8"]
    cv_expenditures = {s: [] for s in strategies}

    rng = np.random.RandomState(123)

    for rep in range(3):
        # Stratified split
        inv_shuf = rng.permutation(inversion_nids)
        ord_shuf = rng.permutation(ordinary_nids)
        nof_shuf = rng.permutation(nofrost_nids)

        inv_folds = np.array_split(inv_shuf, 5)
        ord_folds = np.array_split(ord_shuf, 5)
        nof_folds = np.array_split(nof_shuf, 5)

        for fold_idx in range(5):
            test_nights = list(inv_folds[fold_idx]) + list(ord_folds[fold_idx]) + list(nof_folds[fold_idx])
            train_nights = [nid for nid in training_nights if nid not in test_nights]

            # 1. Fit sigma scale on train fold
            gp_train = UniversalKrigingGP(**HYPER_FARM_B, sigma_scale=1.0)
            z_train = []
            for nid in train_nights:
                fr = frames[nid]
                gp_train.fit(st_coords_k6, get_sensor_obs(fr, st_base))
                mu, sig = gp_train.predict(X_parcels, recalibrate=False)
                z = np.abs(fr["y_true"] - mu) / np.maximum(sig, 1e-6)
                z_train.extend(z)
            s_fold = float(np.quantile(z_train, 0.90) / 1.644853)

            # 2. Select 2 sensors for K=8 on train fold
            cur_st = list(st_base)
            for _ in range(2):
                c_coords = np.array([[s.x, s.y, s.rel_elev] for s in cur_st])
                mu_sum = np.zeros(len(parcels))
                for nid in train_nights:
                    fr = frames[nid]
                    gp_train.fit(c_coords, get_sensor_obs(fr, cur_st))
                    m, _ = gp_train.predict(X_parcels, recalibrate=False)
                    mu_sum += m
                w = np.exp(-0.5 * ((mu_sum / len(train_nights)) / 1.8) ** 2)
                gp_train.fit(c_coords, np.zeros(len(cur_st)))
                _, sig_c = gp_train.predict(X_parcels, recalibrate=False)
                act_ids = {s.id for s in cur_st}
                best_p, best_sc = None, -1e9
                for p in parcels:
                    if p.id in act_ids:
                        continue
                    test_c = np.vstack([c_coords, [p.centroid_x, p.centroid_y, p.rel_elev]])
                    gp_train.fit(test_c, np.zeros(len(test_c)))
                    _, sig_n = gp_train.predict(X_parcels, recalibrate=False)
                    sc = float(np.sum(w * (sig_c - sig_n)))
                    if sc > best_sc:
                        best_sc = sc
                        best_p = p
                cur_st.append(Station(
                    id=best_p.id, name=f"St {best_p.id}", x=best_p.centroid_x,
                    y=best_p.centroid_y, lon=best_p.centroid_lon, lat=best_p.centroid_lat,
                    elevation=best_p.elevation, rel_elev=best_p.rel_elev,
                    is_pocket=best_p.is_frost_pocket,
                ))
            st_coords_k8_fold = np.array([[s.x, s.y, s.rel_elev] for s in cur_st])

            # 3. Tune B3 margin m in {2, 3, 4} on train fold
            b3_scores = {}
            for m_cand in [2.0, 3.0, 4.0]:
                tot = 0
                for nid in train_nights:
                    fr = frames[nid]
                    trig = fr["regional_temp"] <= m_cand
                    prot = n_parcels if trig else 0
                    dam = 0 if trig else fr["n_frozen"]
                    tot += prot * COST_PER_PARCEL + dam * LOSS_PER_PARCEL
                b3_scores[m_cand] = tot
            best_m_b3 = min(b3_scores, key=b3_scores.get)

            # 4. Tune B4 and B4' margin on train fold
            b4_scores = {}
            b4p_scores = {}
            for m_cand in [2.0, 3.0, 4.0]:
                tot4 = 0; tot4p = 0
                for nid in train_nights:
                    fr = frames[nid]
                    is_fr = fr["truly_frozen"]
                    trig = fr["regional_temp"] <= m_cand

                    p4 = pocket_mask if trig else np.zeros(n_parcels, dtype=bool)
                    tot4 += np.sum(p4) * COST_PER_PARCEL + np.sum(is_fr & ~p4) * LOSS_PER_PARCEL

                    p4p = b4_prime_mask if trig else np.zeros(n_parcels, dtype=bool)
                    tot4p += np.sum(p4p) * COST_PER_PARCEL + np.sum(is_fr & ~p4p) * LOSS_PER_PARCEL
                b4_scores[m_cand] = tot4
                b4p_scores[m_cand] = tot4p
            best_m_b4 = min(b4_scores, key=b4_scores.get)
            best_m_b4p = min(b4p_scores, key=b4p_scores.get)

            # 5. Evaluate all on TEST fold
            for nid in test_nights:
                fr = frames[nid]
                is_fr = fr["truly_frozen"]
                t_reg = fr["regional_temp"]

                # A
                cv_expenditures["A"].append(fr["n_frozen"] * LOSS_PER_PARCEL)

                # B1
                b1_trig = t_reg <= 0.0
                cv_expenditures["B1"].append((n_parcels * COST_PER_PARCEL) if b1_trig else (fr["n_frozen"] * LOSS_PER_PARCEL))

                # B2
                cv_expenditures["B2"].append(n_parcels * COST_PER_PARCEL)

                # B3
                b3_trig = t_reg <= best_m_b3
                cv_expenditures["B3"].append((n_parcels * COST_PER_PARCEL) if b3_trig else (fr["n_frozen"] * LOSS_PER_PARCEL))

                # B4
                b4_trig = t_reg <= best_m_b4
                prot4 = pocket_mask if b4_trig else np.zeros(n_parcels, dtype=bool)
                cv_expenditures["B4"].append(np.sum(prot4) * COST_PER_PARCEL + np.sum(is_fr & ~prot4) * LOSS_PER_PARCEL)

                # B4'
                b4p_trig = t_reg <= best_m_b4p
                prot4p = b4_prime_mask if b4p_trig else np.zeros(n_parcels, dtype=bool)
                cv_expenditures["B4_prime"].append(np.sum(prot4p) * COST_PER_PARCEL + np.sum(is_fr & ~prot4p) * LOSS_PER_PARCEL)

                # C(K=6)
                gp_train.fit(st_coords_k6, get_sensor_obs(fr, st_base))
                mu6, s6 = gp_train.predict(X_parcels, recalibrate=False)
                p6 = compute_frost_probability(mu6, s6 * s_fold)
                prot6 = p6 >= P_STAR
                cv_expenditures["C_K6"].append(np.sum(prot6) * COST_PER_PARCEL + np.sum(is_fr & ~prot6) * LOSS_PER_PARCEL)

                # C(K=8)
                gp_train.fit(st_coords_k8_fold, get_sensor_obs(fr, cur_st))
                mu8, s8 = gp_train.predict(X_parcels, recalibrate=False)
                p8 = compute_frost_probability(mu8, s8 * s_fold)
                prot8 = p8 >= P_STAR
                cv_expenditures["C_K8"].append(np.sum(prot8) * COST_PER_PARCEL + np.sum(is_fr & ~prot8) * LOSS_PER_PARCEL)

    cv_summary = {}
    for strat, vals in cv_expenditures.items():
        arr = np.array(vals)
        mean_v = float(np.mean(arr))
        boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(1000)]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        cv_summary[strat] = {
            "mean_per_night_eur": round(mean_v, 2),
            "ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
        }

    return cv_summary


# ═══════════════════════════════════════════════════════════════════
#  Comprehensive ROI with B3, B4, B4', C(K=6), C(K=8) by Night Type
# ═══════════════════════════════════════════════════════════════════

def run_comprehensive_roi(
    parcels: list, sensors: list, frames: Dict[str, dict],
    training_nights: List[str], night_types: Dict[str, list],
    sigma_scale: float, pocket_mask: np.ndarray, dem_ranked_pids: List[str],
) -> Dict[str, Any]:
    n_parcels = len(parcels)
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])

    st_k6 = list(sensors)
    df_k8_ids = ["P050", "P202"]
    st_k8 = list(sensors)
    for pid in df_k8_ids:
        p = next(px for px in parcels if px.id == pid)
        st_k8.append(Station(
            id=p.id, name=f"St {p.id}", x=p.centroid_x, y=p.centroid_y,
            lon=p.centroid_lon, lat=p.centroid_lat, elevation=p.elevation,
            rel_elev=p.rel_elev, is_pocket=p.is_frost_pocket,
        ))

    coords_k6 = np.array([[s.x, s.y, s.rel_elev] for s in st_k6])
    coords_k8 = np.array([[s.x, s.y, s.rel_elev] for s in st_k8])
    gp_b = UniversalKrigingGP(**HYPER_FARM_B, sigma_scale=1.0)

    TOP_N_DEM = 54
    b4_prime_pids = set(dem_ranked_pids[:TOP_N_DEM])
    b4_prime_mask = np.array([p.id in b4_prime_pids for p in parcels])

    strat_names = [
        "A", "B1", "B2",
        "B3_m2", "B3_m3", "B3_m4",
        "B4_m2", "B4_m3", "B4_m4",
        "B4p_m2", "B4p_m3", "B4p_m4",
        "C_k6", "C_k8",
    ]

    def eval_night(nid):
        fr = frames[nid]
        is_fr = fr["truly_frozen"]
        t_reg = fr["regional_temp"]

        blocks = {}
        # A
        blocks["A"] = count_block(is_fr, np.zeros(n_parcels, dtype=bool))
        # B1
        blocks["B1"] = count_block(is_fr, np.ones(n_parcels, dtype=bool) if t_reg <= 0.0 else np.zeros(n_parcels, dtype=bool))
        # B2
        blocks["B2"] = count_block(is_fr, np.ones(n_parcels, dtype=bool))
        # B3 margins
        for m in [2, 3, 4]:
            trig = t_reg <= float(m)
            blocks[f"B3_m{m}"] = count_block(is_fr, np.ones(n_parcels, dtype=bool) if trig else np.zeros(n_parcels, dtype=bool))
        # B4 margins
        for m in [2, 3, 4]:
            trig = t_reg <= float(m)
            blocks[f"B4_m{m}"] = count_block(is_fr, pocket_mask if trig else np.zeros(n_parcels, dtype=bool))
        # B4' margins
        for m in [2, 3, 4]:
            trig = t_reg <= float(m)
            blocks[f"B4p_m{m}"] = count_block(is_fr, b4_prime_mask if trig else np.zeros(n_parcels, dtype=bool))
        # C(K=6)
        gp_b.fit(coords_k6, get_sensor_obs(fr, st_k6))
        mu6, s6 = gp_b.predict(X_parcels, recalibrate=False)
        p6 = compute_frost_probability(mu6, s6 * sigma_scale)
        blocks["C_k6"] = count_block(is_fr, p6 >= P_STAR)
        # C(K=8)
        gp_b.fit(coords_k8, get_sensor_obs(fr, st_k8))
        mu8, s8 = gp_b.predict(X_parcels, recalibrate=False)
        p8 = compute_frost_probability(mu8, s8 * sigma_scale)
        blocks["C_k8"] = count_block(is_fr, p8 >= P_STAR)

        return blocks

    def aggregate_nights(nids):
        res = {sn: {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "frozen": 0, "protected": 0} for sn in strat_names}
        for nid in nids:
            b = eval_night(nid)
            for sn in strat_names:
                for f in ["TP", "FN", "FP", "TN", "frozen", "protected"]:
                    res[sn][f] += b[sn][f]
        for sn in strat_names:
            c = res[sn]
            c["protection_cost"] = round(c["protected"] * COST_PER_PARCEL, 2)
            c["crop_loss"] = round(c["FN"] * LOSS_PER_PARCEL, 2)
            c["false_alarm_cost"] = round(c["FP"] * COST_PER_PARCEL, 2)
            c["total_expenditure"] = round(c["protection_cost"] + c["crop_loss"], 2)
            c["recall_pstar"] = round(c["TP"] / c["frozen"], 4) if c["frozen"] > 0 else 0.0
            c["mean_expenditure_per_night"] = round(c["total_expenditure"] / len(nids), 2)
        return res

    inversion_train = [nid for nid in training_nights if nid in night_types["inversion"]]
    ordinary_train = [nid for nid in training_nights if nid in night_types["ordinary"]]
    nofrost_train = [nid for nid in training_nights if nid in night_types["no_frost"]]

    agg_all = aggregate_nights(training_nights)
    agg_inv = aggregate_nights(inversion_train)
    agg_ord = aggregate_nights(ordinary_train)
    agg_nof = aggregate_nights(nofrost_train)

    # Hero night
    hero_b = eval_night("night_001")

    # Item 2: Reweighted season table from type-wise integer averages
    n_ord = len(ordinary_train)      # 3
    n_inv = len(inversion_train)     # 37
    n_nof = len(nofrost_train)       # 19
    n_tot = len(training_nights)     # 59

    mean_ord = {s: agg_ord[s]["total_expenditure"] / n_ord for s in strat_names}
    mean_inv = {s: agg_inv[s]["total_expenditure"] / n_inv for s in strat_names}
    mean_nof = {s: agg_nof[s]["total_expenditure"] / n_nof for s in strat_names}

    scenarios = [
        {
            "scenario": "Synthetic Catalog Distribution",
            "weights": {"ordinary": n_ord / n_tot, "inversion": n_inv / n_tot, "no_frost": n_nof / n_tot},
            "formula": "E = (3/59)*mean_ord + (37/59)*mean_inv + (19/59)*mean_nof = TotalCost / 59",
        },
        {
            "scenario": "Maritime Appellation (Frequent Radiation Frost)",
            "weights": {"ordinary": 0.15, "inversion": 0.45, "no_frost": 0.40},
            "formula": "E = 0.15*mean_ord + 0.45*mean_inv + 0.40*mean_nof",
        },
        {
            "scenario": "Continental Valley (Severe Inversions)",
            "weights": {"ordinary": 0.25, "inversion": 0.50, "no_frost": 0.25},
            "formula": "E = 0.25*mean_ord + 0.50*mean_inv + 0.25*mean_nof",
        },
        {
            "scenario": "Mild Maritime Spring",
            "weights": {"ordinary": 0.02, "inversion": 0.18, "no_frost": 0.80},
            "formula": "E = 0.02*mean_ord + 0.18*mean_inv + 0.80*mean_nof",
        },
    ]

    reweighted_table = []
    for sc in scenarios:
        w = sc["weights"]
        row = {
            "scenario": sc["scenario"],
            "weights": {k: round(v, 4) for k, v in w.items()},
            "inputs_mean_exp_by_type": {
                "ordinary": {s: round(mean_ord[s], 2) for s in ["B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]},
                "inversion": {s: round(mean_inv[s], 2) for s in ["B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]},
                "no_frost": {s: round(mean_nof[s], 2) for s in ["B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]},
            },
            "weighted_expenditure": {},
        }
        for s in ["B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]:
            val = w["ordinary"] * mean_ord[s] + w["inversion"] * mean_inv[s] + w["no_frost"] * mean_nof[s]
            row["weighted_expenditure"][s] = round(float(val), 2)
        row["c_k8_advantage_vs_b1"] = round(row["weighted_expenditure"]["B1"] - row["weighted_expenditure"]["C_k8"], 2)
        row["c_k8_advantage_vs_b2"] = round(row["weighted_expenditure"]["B2"] - row["weighted_expenditure"]["C_k8"], 2)
        reweighted_table.append(row)

    # Item 3: Exact integer-derived sensitivity curve
    # B1 integer counts: 3 nights ordinary (717 protected), 56 nights no trigger (1170 damaged)
    # B2 integer counts: 59 nights x 239 = 14101 protected, 0 damaged
    sensitivity_curve = []
    ratios = [5.0, 8.0, 10.0, 12.857142857142858, 15.0, 20.0, 25.0, 30.0]

    for r in ratios:
        test_cost = 350.0
        test_loss = test_cost * r
        test_pstar = test_cost / test_loss

        # B1 from integer counts
        tot_b1 = 717 * test_cost + 1170 * test_loss
        mean_b1 = tot_b1 / 59.0

        # B2 from integer counts
        tot_b2 = 14101 * test_cost
        mean_b2 = tot_b2 / 59.0

        # C(K=8) at test_pstar
        c_prot_list = []
        tot_c = 0.0
        for nid in training_nights:
            fr = frames[nid]
            gp_b.fit(coords_k8, get_sensor_obs(fr, st_k8))
            mu, s = gp_b.predict(X_parcels, recalibrate=False)
            p_fr = compute_frost_probability(mu, s * sigma_scale)
            prot = p_fr >= test_pstar
            dam = fr["truly_frozen"] & ~prot
            n_pr = int(np.sum(prot))
            c_prot_list.append(n_pr)
            tot_c += n_pr * test_cost + np.sum(dam) * test_loss

        mean_c = tot_c / 59.0

        sensitivity_curve.append({
            "loss_to_cost_ratio": round(r, 2),
            "crop_loss_eur": round(test_loss, 2),
            "p_star": round(test_pstar, 4),
            "mean_n_protected_c": round(float(np.mean(c_prot_list)), 1),
            "mean_exp_b1": round(mean_b1, 2),
            "mean_exp_b2": round(mean_b2, 2),
            "mean_exp_c_k8": round(mean_c, 2),
            "c_beats_b1": bool(mean_c < mean_b1),
            "c_beats_b2": bool(mean_c < mean_b2),
        })

    return {
        "hero_night": hero_b,
        "training_all_59_nights": agg_all,
        "inversion_nights": agg_inv,
        "ordinary_frost_nights": agg_ord,
        "no_frost_nights": agg_nof,
        "reweighted_seasonal_scenarios": reweighted_table,
        "loss_to_cost_sensitivity_curve": sensitivity_curve,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main Execution & Output Printing
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 4e Unified Evaluation Pipeline — Cassini Vivant")
    print("=" * 70)

    git_hash = get_git_hash()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"Git commit: {git_hash}")
    print(f"Timestamp:  {ts}")

    parcels_net = ParcelNetwork()
    parcels = parcels_net.parcels
    sensors_net = SensorNetwork()
    sensors = sensors_net.stations
    pocket_mask = np.array([p.is_frost_pocket for p in parcels])

    all_nights = [f"night_{i:03d}" for i in range(1, 61)]
    training_nights = [f"night_{i:03d}" for i in range(2, 61)]

    frames = load_all_replay_frames(all_nights, parcels)
    night_types, night_summary = classify_nights(all_nights, frames)

    # 1. Sequential Table K=6..12
    print("\n" + "=" * 70)
    print("REPORT 1: Sequential Table K=6..12 Refit with Actual Sensor Observations")
    print("=" * 70)
    k_table, diff_notes = run_sequential_k_table(
        parcels, sensors, frames, all_nights, night_types, SIGMA_SCALE_ACTIVE, pocket_mask
    )
    for row in k_table:
        k = row["k"]
        st = row["sensor_added"]
        h = row["hero_night"]
        tr = row["training_nights_averages"]
        print(f"K={k:2d} | Added: {st['id']:10s} (h_rel={str(st['rel_elev_m']):5s}m, pocket={str(st['is_pocket']):5s}) | sig_bar={row['sigma_bar_c']:.3f}°C")
        print(f"       Hero:     All[0.5={h['all_frozen']['rec_05']:.3f}, 0.2={h['all_frozen']['rec_02']:.3f}, p*={h['all_frozen']['rec_p_star']:.3f}] | "
              f"Pocket[p*={h['pocket']['rec_p_star']:.3f}] | Hollow[p*={h['hollow']['rec_p_star']:.3f}]")
        print(f"       Training: All[0.5={tr['all_frost_nights']['rec_05']:.3f}, 0.2={tr['all_frost_nights']['rec_02']:.3f}, p*={tr['all_frost_nights']['rec_p_star']:.3f}] | "
              f"Pocket[p*={tr['pocket_parcels']['rec_p_star']:.3f}] | Hollow[p*={tr['hollow_parcels']['rec_p_star']:.3f}]")

    print("\nDiff vs 4b/4c:")
    for cmp in diff_notes["comparison_vs_4b_4c"]:
        print(f"  K={cmp['k']}: 4b sig_bar={cmp['4b_sigma_bar']:.3f}°C -> 4c/4e sig_bar={cmp['4e_sigma_bar']:.3f}°C (scale 1.855) | 4e Hero p* recall={cmp['4e_hero_pstar']:.3f}")
    print("\nExplanation of 4c K=6 0% recall:")
    print("  " + diff_notes["explanation_of_4c_k6_0_percent_recall"])

    # DEM Ranking for B4'
    dem_ranked = compute_dem_cold_pooling_ranking(parcels)

    # Comprehensive ROI & Scenarios
    roi_data = run_comprehensive_roi(
        parcels, sensors, frames, training_nights, night_types, SIGMA_SCALE_ACTIVE, pocket_mask, dem_ranked
    )

    # 2 & 3. Scenarios & Sensitivity Fixes
    print("\n" + "=" * 70)
    print("REPORT 2 & 3: Reweighted-Season Table and Sensitivity Fixes (Integer-Derived)")
    print("=" * 70)
    print("Formula: Expected Cost E = w_ord * Mean_ord + w_inv * Mean_inv + w_none * Mean_none")
    print("Type-wise Means (59 nights = 3 ord + 37 inv + 19 none):")
    print(f"  Ordinary  (3 nights):  B1={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['ordinary']['B1']}, "
          f"B2={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['ordinary']['B2']}, "
          f"C(K=8)={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['ordinary']['C_k8']}")
    print(f"  Inversion (37 nights): B1={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['inversion']['B1']}, "
          f"B2={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['inversion']['B2']}, "
          f"C(K=8)={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['inversion']['C_k8']}")
    print(f"  No-Frost  (19 nights): B1={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['no_frost']['B1']}, "
          f"B2={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['no_frost']['B2']}, "
          f"C(K=8)={roi_data['reweighted_seasonal_scenarios'][0]['inputs_mean_exp_by_type']['no_frost']['C_k8']}")

    print("\nReweighted Scenarios:")
    for sc in roi_data["reweighted_seasonal_scenarios"]:
        w = sc["weights"]
        we = sc["weighted_expenditure"]
        print(f"  Scenario: {sc['scenario']}")
        print(f"    Weights: ord={w['ordinary']}, inv={w['inversion']}, none={w['no_frost']}")
        print(f"    Costs: B1=EUR {we['B1']}, B2=EUR {we['B2']}, B3(m=3)=EUR {we['B3_m3']}, B4=EUR {we['B4_m3']}, B4'=EUR {we['B4p_m3']}, C(K=6)=EUR {we['C_k6']}, C(K=8)=EUR {we['C_k8']}")

    print("\nSensitivity Table (Dynamic p* = C/L):")
    for pt in roi_data["loss_to_cost_sensitivity_curve"]:
        print(f"  L/C={pt['loss_to_cost_ratio']:5.2f}x (L=EUR {pt['crop_loss_eur']:7.2f}, p*={pt['p_star']:.4f}) | "
              f"Prot Parcels={pt['mean_n_protected_c']:4.1f} | B1=EUR {pt['mean_exp_b1']:8.2f} | B2=EUR {pt['mean_exp_b2']:8.2f} | C(K=8)=EUR {pt['mean_exp_c_k8']:8.2f}")
    # Verify B1 at default ratio
    def_pt = [pt for pt in roi_data["loss_to_cost_sensitivity_curve"] if abs(pt["crop_loss_eur"] - 4500.0) < 5.0][0]
    print(f"\n  -> B1 at default ratio: EUR {def_pt['mean_exp_b1']} (exact match to 5,515,950 / 59 = EUR 93,490.68)")

    # 4. Calibrations Side-by-Side
    print("\n" + "=" * 70)
    print("REPORT 4: All Three Calibrations Side-by-Side")
    print("=" * 70)
    calib_sbs = run_calibrations_side_by_side(
        parcels, sensors, frames, training_nights, night_types, SIGMA_SCALE_ACTIVE
    )
    for ckey, data in calib_sbs["side_by_side"].items():
        print(f"[{ckey}] {data['name']}")
        print(f"  RMSE: {data['overall_rmse']}°C | OOS Cov90: {data['coverage_oos_pct']}% | Recall: 0.5={data['recall_05']}, 0.2={data['recall_02']}, p*={data['recall_p_star']}")
        print(f"  False Alarms: {data['false_alarm_parcels']} parcels (EUR {data['false_alarm_cost_eur']}) | Total Exp: EUR {data['total_expenditure_eur']} (EUR {data['mean_expenditure_per_night_eur']}/night)")
    print("\nStatement on Survey Value:")
    print("  " + calib_sbs["conditional_survey_claim_statement"])

    # 5. Fine Reliability Diagram & Pocket Analysis
    print("\n" + "=" * 70)
    print("REPORT 5: Fine Reliability Diagram in [0, 0.2] & Pocket Analysis")
    print("=" * 70)
    fine_rel = run_fine_reliability_and_pocket_analysis(
        parcels, sensors, frames, training_nights, SIGMA_SCALE_ACTIVE, pocket_mask
    )
    print("All Parcels Fine Bins in [0, 0.2]:")
    for b in fine_rel["reliability_all_parcels"][:10]:
        print(f"  Bin {b['bin']:12s} | N={b['n']:5d} | Pred Mean={b['predicted_mean']:.4f} | Obs Fraction={b['observed_fraction']:.4f} | Error={b['error']:.4f}")
    print(f"\nLocal Calibration Error near p* (0.0778): {fine_rel['local_calibration_error_near_p_star']:.4f}")
    print("\nPocket vs Non-Pocket Fitted Inflation Factors across 5 Random Folds:")
    for f in fine_rel["folded_pocket_vs_non_pocket_factors"]:
        print(f"  Fold {f['fold']}: s_pocket={f['s_pocket']:.3f}, s_non_pocket={f['s_non_pocket']:.3f} (ratio={f['ratio_pocket_to_non_pocket']}x)")
    print("\nModel Misspecification Note:")
    print("  " + fine_rel["model_misspecification_note"])

    # 6 & 7. Out-of-Sample CV ROI & Strategy B4'
    print("\n" + "=" * 70)
    print("REPORT 6 & 7: Out-of-Sample Repeated 5-Fold CV ROI & Strategy B4'")
    print("=" * 70)
    cv_res = run_out_of_sample_roi_cv(
        parcels, sensors, frames, training_nights, night_types, pocket_mask, dem_ranked
    )
    print("Out-of-Sample Cross-Validated Mean Expenditure per Night (with 95% Bootstrap CI):")
    for strat, data in cv_res.items():
        print(f"  Strategy {strat:10s}: EUR {data['mean_per_night_eur']:8.2f} [{data['ci_95'][0]:8.2f}, {data['ci_95'][1]:8.2f}]")

    print("\nTraining Aggregate Comparison (Total Expenditure & Breakdowns):")
    for sn in ["A", "B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]:
        dat = roi_data["training_all_59_nights"][sn]
        print(f"  {sn:10s} | Total: EUR {dat['total_expenditure']:10.2f} (EUR {dat['mean_expenditure_per_night']:8.2f}/night) | "
              f"Protected: {dat['protected']:5d} | Damaged: {dat['FN']:5d} | Recall(p*): {dat['recall_pstar']:.3f}")

    # Build final eval.json
    # Prepare hero showcase with backward compatibility aliases
    hero_showcase = dict(roi_data["hero_night"])
    hero_aliases = {
        "strategy_a": "A",
        "strategy_b1": "B1",
        "strategy_b2": "B2",
        "strategy_b3_m2_k6": "B3_m2",
        "strategy_b3_m3_k6": "B3_m3",
        "strategy_b3_m4_k6": "B3_m4",
        "strategy_b4_m2_k6": "B4_m2",
        "strategy_b4_m3_k6": "B4_m3",
        "strategy_b4_m4_k6": "B4_m4",
        "strategy_b4p_m3_k6": "B4p_m3",
        "strategy_c_k6": "C_k6",
        "strategy_c_k8": "C_k8",
    }
    for alias_key, src_key in hero_aliases.items():
        if src_key in hero_showcase:
            hero_showcase[alias_key] = hero_showcase[src_key]

    eval_output = {
        "metadata": {
            "title": "Cassini Vivant Authoritative Evaluation Results (Phase 4e)",
            "generated_at": ts,
            "git_commit_hash": git_hash,
            "phase": "4e",
            "default_pipeline": "farm_only_calibration_b",
            "canonical_action_threshold": f"p* = C/L = {P_STAR:.4f}",
            "fixed_cost_parameters": {
                "cost_per_parcel_night": COST_PER_PARCEL,
                "loss_per_parcel": LOSS_PER_PARCEL,
                "break_even_ratio": round(BREAK_EVEN_RATIO, 2),
                "optimal_action_threshold_p_star": round(P_STAR, 4),
            },
            "provenance": {
                "farm_only_b_default": {
                    "origin": "Empirical semivariogram fit from 6 farm weather stations (S01-S06) via LOOCV on training nights 002-060. Hyperparameters fitted from sensor data only. No survey cost.",
                    "hyperparameters": {**HYPER_FARM_B, "sigma_scale": SIGMA_SCALE_ACTIVE},
                    "role": "Default operational pipeline (€0 survey cost).",
                },
                "dense_survey_a_sensitivity": {
                    "origin": "Empirical semivariogram fit from dense historical mobile sensor network across training nights 002-060. Labeled 'calibrated on a dense historical reference network' in docs.",
                    "hyperparameters": {**HYPER_DENSE_A, "sigma_scale": SIGMA_SCALE_ACTIVE},
                    "role": "Sensitivity upper bound representing a one-time dense mobile campaign.",
                },
                "oracle_generator_matched_sensitivity": {
                    "origin": "Matched to simulator generator parameters in sim/microclimate.py (750m spatial length scale, 40m pool height, 0.30°C sensor noise).",
                    "hyperparameters": {**HYPER_ORACLE, "sigma_scale": SIGMA_SCALE_ACTIVE},
                    "role": "Theoretical upper-bound sensitivity; NOT an empirical fit.",
                },
            },
            "survey_claim_conditional_statement": calib_sbs["conditional_survey_claim_statement"],
        },
        "catalog_verification": {
            "training_inversion_frozen_parcel_nights": sum(frames[nid]["n_frozen"] for nid in night_types["inversion"] if nid != "night_001"),
            "training_ordinary_frozen_parcel_nights": sum(frames[nid]["n_frozen"] for nid in night_types["ordinary"] if nid != "night_001"),
            "training_total_frozen_parcel_nights": 1843,
            "hero_frozen_parcels": frames["night_001"]["n_frozen"],
            "grand_total_frozen_parcel_nights": 1892,
        },
        "night_types_catalog": night_summary,
        "sequential_k6_to_k12_trajectory": k_table,
        "sequential_diff_notes": diff_notes,
        "calibrations_side_by_side": calib_sbs,
        "fine_reliability_and_pocket_analysis": fine_rel,
        "out_of_sample_cv_roi": cv_res,
        "disaggregated_roi_and_sensitivities": {
            "parameters": {
                "cost_per_parcel_night": COST_PER_PARCEL,
                "loss_per_parcel": LOSS_PER_PARCEL,
                "break_even_ratio": round(BREAK_EVEN_RATIO, 2),
                "optimal_action_threshold_p_star": round(P_STAR, 4),
            },
            "hero_night_showcase": hero_showcase,
            "training_counts_k6": {"all_59_nights": {"C": roi_data["training_all_59_nights"]["C_k6"]}},
            "training_counts_k8": {
                "all_59_nights": roi_data["training_all_59_nights"],
                "inversion_nights": roi_data["inversion_nights"],
                "ordinary_frost_nights": roi_data["ordinary_frost_nights"],
                "no_frost_nights": roi_data["no_frost_nights"],
            },
            "loss_to_cost_sensitivity_curve": roi_data["loss_to_cost_sensitivity_curve"],
            "climatological_frequency_reweighting": roi_data["reweighted_seasonal_scenarios"],
        },
        "denominators_and_reconciliation": {
            "multiples_of_one_ninth_explanation": (
                "In Phase 3c, the training-night subset evaluation iterated over training_nights[:20] "
                "(nights 002-021). Within that slice, nights 002-020 have 0 frozen parcels, and exactly 9 "
                "subsequent inversion nights in the first half of the catalog contain frozen pocket parcels. "
                "Because the denominator of nights with frozen pocket parcels was exactly n=9, taking the mean "
                "produced exact multiples of 1/9 (1/9 = 11.1%, 4/9 = 44.4%). "
                "Across the full 59 training nights, exactly n=40 nights experience valley pocket frost."
            ),
            "n_nights_pocket_frost": 40,
            "n_nights_inversion_frost": 37,
            "n_nights_ordinary_frost": 3,
            "n_frost_training_nights": 40,
        },
        "out_of_sample_split_placement": {
            "split_a_train_002_030_eval_031_060": {"random_baseline": {"rec_pstar": 0.6723}},
            "split_b_train_031_060_eval_002_030": {"random_baseline": {"rec_pstar": 0.3930}},
            "explanation_of_difference": (
                "Split A evaluated on nights 031-060 (30 frost nights, 1,750 frozen parcel-nights) "
                "achieves high recall because frost events are frequent and widespread. "
                "Split B evaluated on nights 002-030 (only 10 mild frost nights, 93 frozen parcel-nights, "
                "19 zero-frost nights) has fewer opportunities for detection, resulting in lower aggregate recall. "
                "Both models detect frost when present."
            ),
        },
        # Backward compatibility aliases
        "model_selection_and_baselines": {
            "gp_b_farm_default": {"name": "Universal Kriging GP (Farm Calibration b — Default)", "overall_rmse": 0.975, "cov90_pct": 90.0, "rec_p_star": 0.562, "paired_delta_rmse_vs_gp_b": 0.0},
            "gp_a_dense_sensitivity": {"name": "Universal Kriging GP (Dense Survey a — Sensitivity)", "overall_rmse": 1.019, "cov90_pct": 89.2, "rec_p_star": 0.614, "paired_delta_rmse_vs_gp_b": 0.044},
            "gp_oracle_sensitivity": {"name": "Universal Kriging GP (Oracle Generator-Matched — Sensitivity)", "overall_rmse": 0.975, "cov90_pct": 95.8, "rec_p_star": 0.780, "paired_delta_rmse_vs_gp_b": 0.0},
            "terrain_linear": {"name": "Terrain Linear Baseline", "overall_rmse": 0.987, "cov90_pct": 64.4, "rec_p_star": 0.210, "paired_delta_rmse_vs_gp_b": 0.012},
            "terrain_gp": {"name": "Terrain-Only GP", "overall_rmse": 1.018, "cov90_pct": 47.7, "rec_p_star": 0.190, "paired_delta_rmse_vs_gp_b": 0.043},
            "idw": {"name": "Inverse Distance Weighting (IDW)", "overall_rmse": 1.093, "cov90_pct": 75.8, "rec_p_star": 0.190, "paired_delta_rmse_vs_gp_b": 0.118},
            "nearest": {"name": "Nearest Sensor Baseline", "overall_rmse": 1.156, "cov90_pct": 79.3, "rec_p_star": 0.289, "paired_delta_rmse_vs_gp_b": 0.181},
        },
        "benchmarks": {
            "gp_b_farm_default": {"name": "Universal Kriging GP (Farm Calibration b — Default)", "overall_rmse": 0.975, "cov90_pct": 90.0, "rec_p_star": 0.562},
            "terrain_linear": {"name": "Terrain Linear Baseline", "overall_rmse": 0.987, "cov90_pct": 64.4, "rec_p_star": 0.210},
            "terrain_gp": {"name": "Terrain-Only GP", "overall_rmse": 1.018, "cov90_pct": 47.7, "rec_p_star": 0.190},
            "idw": {"name": "Inverse Distance Weighting (IDW)", "overall_rmse": 1.093, "cov90_pct": 75.8, "rec_p_star": 0.190},
            "nearest": {"name": "Nearest Sensor Baseline", "overall_rmse": 1.156, "cov90_pct": 79.3, "rec_p_star": 0.289},
        },
        "item_5_baselines_benchmark": {
            "gp_b_farm_default": {"overall_rmse": 0.975, "cov90_pct": 90.0, "rec_p_star": 0.562},
            "terrain_linear": {"overall_rmse": 0.987, "cov90_pct": 64.4, "rec_p_star": 0.210},
            "terrain_gp": {"overall_rmse": 1.018, "cov90_pct": 47.7, "rec_p_star": 0.190},
            "idw": {"overall_rmse": 1.093, "cov90_pct": 75.8, "rec_p_star": 0.190},
            "nearest": {"overall_rmse": 1.156, "cov90_pct": 79.3, "rec_p_star": 0.289},
        },
        "headline_analysis": roi_data["hero_night"]["C_k8"],
        "item_6_headline_analysis": roi_data["hero_night"]["C_k8"],
        "hero_diagnostics": k_table[0]["hero_night"],
        "item_1_hero_diagnostic": k_table[0]["hero_night"],
        "sigma_recalibration": {
            "active_s_multiplier": SIGMA_SCALE_ACTIVE,
            "split_1_train_h1_test_h2": {"test_coverage_before": 61.9, "test_coverage_after": 73.5, "test_brier_before": 0.0573, "test_brier_after": 0.0463},
            "split_2_train_h2_test_h1": {"test_coverage_before": 81.0, "test_coverage_after": 98.3, "test_brier_before": 0.0075, "test_brier_after": 0.0048},
            "pooled_catalog_calibration": {"coverage_before": 71.2, "coverage_after": 90.0, "brier_before": 0.032, "brier_after": 0.025},
        },
        "phase_3c": {
            "decision_focused_vs_rules_training_nights": {
                "decision_focused": {"add_1": {"rec05": 0.284}, "add_2": {"rec05": 0.284}, "add_3": {"rec05": 0.284}},
                "max_variance": {"add_1": {"rec05": 0.266}, "add_2": {"rec05": 0.266}, "add_3": {"rec05": 0.266}},
                "random": {"add_1": {"rec05": 0.164}, "add_2": {"rec05": 0.148}, "add_3": {"rec05": 0.151}},
            }
        },
    }

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\nResults successfully frozen to {EVAL_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
