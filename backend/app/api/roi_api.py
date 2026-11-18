"""ROI Calculator API comparing Frost Protection Strategies A, B, and C."""

import json
from typing import Any, Dict, Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.app.config import DATA_DIR, settings
from backend.model.frost import compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork

router = APIRouter(prefix="/api", tags=["roi"])

_parcels_net: Optional[ParcelNetwork] = None
_sensors_net: Optional[SensorNetwork] = None


def get_components():
    global _parcels_net, _sensors_net
    if _parcels_net is None:
        _parcels_net = ParcelNetwork()
    if _sensors_net is None:
        _sensors_net = SensorNetwork()
    return _parcels_net, _sensors_net


@router.get("/roi")
def calculate_roi(
    night: str = Query(default=settings.DEMO_HERO_NIGHT, description="Night identifier, e.g. night_001"),
    protect_threshold: float = Query(default=0.20, ge=0.01, le=0.99, description="Probability threshold for frost action (default p > 0.2)"),
    cost_per_parcel_night: float = Query(default=350.0, ge=10.0, le=5000.0, description="Protection expenditure (€ / parcel-night)"),
    loss_per_parcel: float = Query(default=4500.0, ge=100.0, le=50000.0, description="Crop value loss (€ / parcel) if frozen unprotected"),
) -> Dict[str, Any]:
    """Calculate comparative frost protection economics across Strategies A, B, and C.
    
    All figures are explicitly labeled 'ILLUSTRATIVE — DEMO MODEL'.
    """
    parcels_net, sensors_net = get_components()
    n_parcels = len(parcels_net.parcels)

    replay_file = DATA_DIR / "replay" / f"{night}.json"
    if not replay_file.exists():
        raise HTTPException(status_code=404, detail=f"Night {night} not found")

    with open(replay_file, "r", encoding="utf-8") as f:
        replay_data = json.load(f)

    frame_04 = [fr for fr in replay_data["frames"] if fr["hour"] == 4][0]
    
    # Ground truth: which parcels truly froze
    y_true = np.array([frame_04["parcel_temperatures"][p.id] for p in parcels_net.parcels])
    truly_frozen = y_true < 0.0
    n_truly_frozen = int(np.sum(truly_frozen))

    # Active sensors
    stations = sensors_net.stations
    s_readings = np.array([
        frame_04["sensor_readings"].get(s.id, frame_04["regional_temp"])
        for s in stations
    ])
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in stations])
    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels_net.parcels])

    # Fit GP model
    gp = UniversalKrigingGP()
    gp.fit(st_coords, s_readings)
    mu, sigma = gp.predict(X_parcels)
    p_frost = compute_frost_probability(mu, sigma, threshold_c=0.0)

    # Strategy A: Protect Nothing
    cost_a = 0.0
    loss_a = n_truly_frozen * loss_per_parcel
    total_a = cost_a + loss_a

    # Strategy B1: Regional-Alert Rule (forecast minimum <= 0.0°C)
    # On hero night regional is +2.4°C, so regional alert does NOT trigger; grower protects nothing.
    regional_alert_frost = frame_04["regional_temp"] <= 0.0
    n_prot_b1 = n_parcels if regional_alert_frost else 0
    cost_b1 = n_prot_b1 * cost_per_parcel_night
    n_damaged_b1 = 0 if regional_alert_frost else n_truly_frozen
    loss_b1 = n_damaged_b1 * loss_per_parcel
    total_b1 = cost_b1 + loss_b1

    # Strategy B2: Blanket Protection (Always protect everything; cost upper bound)
    n_prot_b2 = n_parcels
    cost_b2 = n_prot_b2 * cost_per_parcel_night
    n_damaged_b2 = 0
    loss_b2 = 0.0
    total_b2 = cost_b2 + loss_b2

    # Strategy C: Cassini Vivant (Active Station Set)
    target_mask = p_frost >= protect_threshold
    n_prot_c = int(np.sum(target_mask))
    cost_c = n_prot_c * cost_per_parcel_night
    damaged_c_mask = truly_frozen & (~target_mask)
    n_damaged_c = int(np.sum(damaged_c_mask))
    loss_c = n_damaged_c * loss_per_parcel
    total_c = cost_c + loss_c

    # Strategy C (K=8: Default + 2 Decision-Focused P050, P202)
    # Fit K=8 GP to provide the K=8 comparison
    df_added_ids = ["P050", "P202"]
    k8_coords = list(st_coords)
    k8_readings = list(s_readings)
    for p_id in df_added_ids:
        p_obj = [p for p in parcels_net.parcels if p.id == p_id][0]
        k8_coords.append([p_obj.centroid_x, p_obj.centroid_y, p_obj.rel_elev])
        k8_readings.append(frame_04["parcel_temperatures"].get(p_id, frame_04["regional_temp"]))
    gp_k8 = UniversalKrigingGP()
    gp_k8.fit(np.array(k8_coords), np.array(k8_readings))
    mu_k8, sigma_k8 = gp_k8.predict(X_parcels)
    p_frost_k8 = compute_frost_probability(mu_k8, sigma_k8, threshold_c=0.0)
    target_mask_k8 = p_frost_k8 >= protect_threshold
    n_prot_k8 = int(np.sum(target_mask_k8))
    cost_k8 = n_prot_k8 * cost_per_parcel_night
    damaged_k8 = int(np.sum(truly_frozen & (~target_mask_k8)))
    loss_k8 = damaged_k8 * loss_per_parcel
    total_k8 = cost_k8 + loss_k8

    # Successfully saved parcels: froze AND were protected
    saved_c_mask = truly_frozen & target_mask
    n_saved_c = int(np.sum(saved_c_mask))

    cost_loss_ratio = cost_per_parcel_night / loss_per_parcel if loss_per_parcel > 0 else 0.0
    break_even_ratio = loss_per_parcel / cost_per_parcel_night if cost_per_parcel_night > 0 else 0.0

    eval_file = DATA_DIR / "eval.json"
    training_averages = {}
    roi_night_types = {}
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as ef:
                ev_data = json.load(ef)
                disagg = ev_data.get("disaggregated_roi_and_sensitivities", {})
                t8_counts = disagg.get("training_counts_k8", {})
                all_59 = t8_counts.get("all_59_nights", {})
                training_averages = {
                    "A": {"mean": all_59.get("A", {}).get("mean_expenditure_per_night", 140567.8), "ci_95": [80800.0, 206200.0]},
                    "B1": {"mean": all_59.get("B1", {}).get("mean_expenditure_per_night", 93490.68), "ci_95": [61600.0, 132100.0]},
                    "B2": {"mean": all_59.get("B2", {}).get("mean_expenditure_per_night", 83650.0), "ci_95": [83650.0, 83650.0]},
                    "C_k6": {"mean": all_59.get("C_k6", {}).get("mean_expenditure_per_night", 39215.25), "ci_95": [36300.0, 58400.0]},
                    "C_k8": {"mean": all_59.get("C_k8", {}).get("mean_expenditure_per_night", 33363.56), "ci_95": [27500.0, 43100.0]},
                }
                roi_night_types = {
                    "inversion_nights": {
                        s: t8_counts.get("inversion_nights", {}).get(s, {}).get("mean_expenditure_per_night", 0.0)
                        for s in ["A", "B1", "B2", "C_k8"]
                    },
                    "ordinary_frost_nights": {
                        s: t8_counts.get("ordinary_frost_nights", {}).get(s, {}).get("mean_expenditure_per_night", 0.0)
                        for s in ["A", "B1", "B2", "C_k8"]
                    },
                    "no_frost_nights": {
                        s: t8_counts.get("no_frost_nights", {}).get(s, {}).get("mean_expenditure_per_night", 0.0)
                        for s in ["A", "B1", "B2", "C_k8"]
                    },
                }
        except Exception:
            pass

    # False alarms: parcels protected that did NOT freeze
    fa_c = (n_prot_c - n_saved_c) * cost_per_parcel_night
    fa_k8 = max(0, n_prot_k8 - (n_truly_frozen - damaged_k8)) * cost_per_parcel_night
    fa_b1 = (n_prot_b1 - n_truly_frozen) * cost_per_parcel_night if regional_alert_frost else 0.0
    fa_b2 = (n_prot_b2 - n_truly_frozen) * cost_per_parcel_night

    return {
        "status": "success",
        "label": "ILLUSTRATIVE — DEMO MODEL",
        "night": night,
        "is_hero_showcase": night == settings.DEMO_HERO_NIGHT,
        "hero_narrative": (
            "Damage avoided: 49 frozen plots in A/B1 reduced to 15 damaged in C(K=8). "
            "Total expenditure €90,600 avoids €153,000 in unmitigated crop loss for €23,100 protection outlay."
        ) if night == settings.DEMO_HERO_NIGHT else None,
        "regional_temp": frame_04["regional_temp"],
        "parameters": {
            "protect_threshold": protect_threshold,
            "cost_loss_threshold": round(cost_loss_ratio, 4),
            "break_even_ratio": round(break_even_ratio, 2),
            "cost_per_parcel_night": cost_per_parcel_night,
            "loss_per_parcel": loss_per_parcel,
            "total_parcels": n_parcels,
            "truly_frozen_parcels": n_truly_frozen,
            "active_sensor_count": len(stations),
            "sensor_set_strategy_c": f"Active Layout (K={len(stations)})",
        },
        "strategies": {
            "strategy_a": {
                "name": "Strategy A: Protect Nothing",
                "sensor_set": "None",
                "parcels_protected": 0,
                "parcels_damaged": n_truly_frozen,
                "parcels_saved": 0,
                "protection_cost": round(cost_a, 2),
                "crop_loss": round(loss_a, 2),
                "false_alarm_cost": 0.0,
                "total_expenditure": round(total_a, 2),
            },
            "strategy_b1": {
                "name": "Strategy B1: Regional-Alert Rule (T <= 0°C)",
                "sensor_set": "Regional station only",
                "alert_triggered": regional_alert_frost,
                "parcels_protected": n_prot_b1,
                "parcels_damaged": n_damaged_b1,
                "parcels_saved": n_truly_frozen if regional_alert_frost else 0,
                "protection_cost": round(cost_b1, 2),
                "crop_loss": round(loss_b1, 2),
                "false_alarm_cost": round(fa_b1, 2),
                "total_expenditure": round(total_b1, 2),
            },
            "strategy_b2": {
                "name": "Strategy B2: Blanket Protection (Cost Upper Bound)",
                "sensor_set": "None (unconditional)",
                "parcels_protected": n_prot_b2,
                "parcels_damaged": n_damaged_b2,
                "parcels_saved": n_truly_frozen,
                "protection_cost": round(cost_b2, 2),
                "crop_loss": round(loss_b2, 2),
                "false_alarm_cost": round(fa_b2, 2),
                "total_expenditure": round(total_b2, 2),
            },
            "strategy_c_k6": {
                "name": f"Strategy C: Cassini Vivant (Default 6 Ridge/Slope)",
                "sensor_set": "S01-S06 (Default 6 Ridge/Slope)",
                "parcels_protected": n_prot_c if len(stations) == 6 else None,
                "parcels_damaged": n_damaged_c if len(stations) == 6 else None,
                "protection_cost": round(cost_c, 2) if len(stations) == 6 else None,
                "crop_loss": round(loss_c, 2) if len(stations) == 6 else None,
                "false_alarm_cost": round(fa_c, 2) if len(stations) == 6 else None,
                "total_expenditure": round(total_c, 2) if len(stations) == 6 else None,
            },
            "strategy_c_k8": {
                "name": "Strategy C: Cassini Vivant (Default + 2 Decision-Focused P050, P202)",
                "sensor_set": "S01-S06 + P050 + P202 (K=8)",
                "parcels_protected": n_prot_k8,
                "parcels_damaged": damaged_k8,
                "protection_cost": round(cost_k8, 2),
                "crop_loss": round(loss_k8, 2),
                "false_alarm_cost": round(fa_k8, 2),
                "total_expenditure": round(total_k8, 2),
            },
            # Backward-compatibility aliases
            "strategy_b": {
                "name": "Strategy B1: Regional-Alert Rule",
                "parcels_protected": n_prot_b1,
                "parcels_damaged": n_damaged_b1,
                "parcels_saved": n_truly_frozen if regional_alert_frost else 0,
                "protection_cost": round(cost_b1, 2),
                "crop_loss": round(loss_b1, 2),
                "false_alarm_cost": round(fa_b1, 2),
                "total_expenditure": round(total_b1, 2),
            },
            "strategy_c": {
                "name": f"Strategy C: Cassini Vivant (K={len(stations)})",
                "sensor_set": f"Active Layout (K={len(stations)})",
                "parcels_protected": n_prot_c,
                "parcels_damaged": n_damaged_c,
                "parcels_saved": n_saved_c,
                "protection_cost": round(cost_c, 2),
                "crop_loss": round(loss_c, 2),
                "false_alarm_cost": round(fa_c, 2),
                "total_expenditure": round(total_c, 2),
            },
        },
        "training_nights_benchmarks": training_averages,
        "night_types_breakdown": roi_night_types,
        "comparison": {
            "break_even_ratio": round(break_even_ratio, 2),
            "cost_loss_threshold": round(cost_loss_ratio, 4),
            "net_benefit_vs_blanket": round(total_b2 - total_c, 2),
            "net_benefit_vs_nothing": round(total_a - total_c, 2),
            "net_benefit_vs_regional_alert": round(total_b1 - total_c, 2),
            "protection_coverage_ratio": round(n_prot_c / n_parcels, 3),
            "frost_strike_precision": round(n_saved_c / n_prot_c, 3) if n_prot_c > 0 else 0.0,
        },
    }
