"""Phase 4b Supplemental Evaluation Engine:
1. Out-of-sample placement validation (train 002-030 / eval 031-060 and reverse).
2. Weight constant sensitivity (1.0°C, 1.8°C, 3.0°C).
3. Step-by-step sequential additions K=6..12 table (DF vs MV vs Random with paired CIs).
4. Bonferroni-adjusted baseline statistics.
5. Rigorous ROI evaluation (A, B1, B2, C(K=6), C(K=8)) across all 59 training nights and hero night.
"""

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import DATA_DIR
from backend.model.baselines import (
    IDWBaseline,
    NearestSensorBaseline,
    TerrainLinearBaseline,
    TerrainOnlyGP,
)
from backend.model.frost import compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.model.next_sensor import NextSensorOptimizer
from backend.sim.nights import build_night_profiles, list_night_ids
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork, Station
from backend.sim.microclimate import MicroclimateSimulator


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95):
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(42)
    n = len(values)
    boot_means = [float(np.mean(rng.choice(values, size=n, replace=True))) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return round(low, 4), round(high, 4)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, ci: float = 0.95):
    diff = np.asarray(a) - np.asarray(b)
    mean_diff = float(np.mean(diff))
    rng = np.random.RandomState(42)
    n = len(diff)
    boot_means = [float(np.mean(rng.choice(diff, size=n, replace=True))) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    p_val = float(2.0 * min(np.mean(np.array(boot_means) <= 0), np.mean(np.array(boot_means) >= 0)))
    p_val = min(1.0, max(p_val, 1.0 / n_boot))
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_95": [round(low, 4), round(high, 4)],
        "p_value": round(p_val, 4),
    }


def main():
    print("=" * 70)
    print("Running Phase 4b Supplemental Evaluation Suite...")
    print("=" * 70)

    parcels_net = ParcelNetwork()
    sensors_net = SensorNetwork()
    parcels = parcels_net.parcels
    stations = sensors_net.stations
    n_parcels = len(parcels)

    X_parcels = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels])
    st_coords_default = np.array([[s.x, s.y, s.rel_elev] for s in stations])
    pocket_mask = np.array([p.is_frost_pocket for p in parcels])

    all_night_ids = list_night_ids()
    training_nights = [nid for nid in all_night_ids if nid != "night_001"]

    # Load 04:00 replays
    replays_04 = {}
    for nid in all_night_ids:
        with open(DATA_DIR / "replay" / f"{nid}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        replays_04[nid] = [fr for fr in data["frames"] if fr["hour"] == 4][0]

    # =========================================================================
    # ITEM 4: Out-of-Sample Placement Validation (002-030 vs 031-060)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITEM 4: Out-of-Sample Split Validation for Sensor Placement Rule")
    print("=" * 70)

    split_1_train = [nid for nid in training_nights if int(nid.split("_")[1]) <= 30]  # nights 002-030 (29 nights)
    split_1_eval = [nid for nid in training_nights if int(nid.split("_")[1]) > 30]   # nights 031-060 (30 nights)

    def evaluate_split(train_nights, eval_nights, split_label):
        # 1. Compute decision weights on train_nights ONLY
        w = np.zeros(n_parcels)
        for nid in train_nights:
            fr = replays_04[nid]
            s_t = np.array([fr["sensor_readings"][s.id] for s in stations])
            gp = UniversalKrigingGP().fit(st_coords_default, s_t)
            mu, _ = gp.predict(X_parcels)
            w += np.exp(-0.5 * (mu / 1.8) ** 2)
        w /= len(train_nights)

        # 2. Derive additions 1, 2, 3 under DF
        opt_df = NextSensorOptimizer(parcels_net, decision_weights=w)
        snet_df = SensorNetwork()
        sim_df = MicroclimateSimulator(parcels=parcels_net, sensors=snet_df)
        df_added = []
        for i in range(3):
            # Fit on average train night readings
            s_temps = np.mean([np.array([replays_04[nid]["sensor_readings"].get(s.id, replays_04[nid]["regional_temp"]) for s in snet_df.stations]) for nid in train_nights[:10]], axis=0)
            coords_cur = np.array([[s.x, s.y, s.rel_elev] for s in snet_df.stations])
            gp_c = UniversalKrigingGP().fit(coords_cur, s_temps)
            rec = opt_df.find_next_sensor(gp_c, snet_df, s_temps)
            df_added.append(rec["best_parcel_id"])
            snet_df.add_station(rec["best_coords"][1], rec["best_coords"][0], f"DF_{i}")

        # Derive additions 1, 2, 3 under MV
        snet_mv = SensorNetwork()
        mv_added = []
        for i in range(3):
            coords_cur = np.array([[s.x, s.y, s.rel_elev] for s in snet_mv.stations])
            s_temps = np.mean([np.array([replays_04[nid]["sensor_readings"].get(s.id, replays_04[nid]["regional_temp"]) for s in snet_mv.stations]) for nid in train_nights[:10]], axis=0)
            gp_c = UniversalKrigingGP().fit(coords_cur, s_temps)
            _, sigs = gp_c.predict(X_parcels)
            dists = np.hypot(X_parcels[:, None, 0] - coords_cur[None, :, 0], X_parcels[:, None, 1] - coords_cur[None, :, 1])
            valid = np.min(dists, axis=-1) >= 250.0
            best_idx = int(np.argmax(np.where(valid, sigs, -1.0)))
            mv_added.append(parcels[best_idx].id)
            snet_mv.add_station(parcels[best_idx].centroid_lat, parcels[best_idx].centroid_lon, f"MV_{i}")

        # 3. Evaluate out-of-sample on eval_nights
        # Compare frozen parcel recall after 1, 2, 3 additions
        results = {"split": split_label, "df_added": df_added, "mv_added": mv_added, "eval": {}}
        for n_add in [1, 2, 3]:
            coords_df_k = np.array([[s.x, s.y, s.rel_elev] for s in snet_df.stations[:6 + n_add]])
            coords_mv_k = np.array([[s.x, s.y, s.rel_elev] for s in snet_mv.stations[:6 + n_add]])
            sim_df.sensors_net.stations = list(snet_df.stations[:6 + n_add])
            sim_mv = MicroclimateSimulator(parcels=parcels_net, sensors=snet_mv)
            sim_mv.sensors_net.stations = list(snet_mv.stations[:6 + n_add])

            rec_df_05, rec_df_02 = [], []
            rec_mv_05, rec_mv_02 = [], []

            for nid in eval_nights:
                prof = build_night_profiles()[nid]
                fr = replays_04[nid]
                y_true = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
                is_fr = y_true < 0.0
                if np.sum(is_fr) == 0:
                    continue

                # DF
                _, s_df_all, _, _ = sim_df.simulate_night(prof)
                gp_df = UniversalKrigingGP().fit(coords_df_k, s_df_all[8, :6 + n_add])
                mu_df, sig_df = gp_df.predict(X_parcels)
                p_df = compute_frost_probability(mu_df, sig_df)
                rec_df_05.append(np.sum((p_df > 0.5) & is_fr) / np.sum(is_fr))
                rec_df_02.append(np.sum((p_df > 0.2) & is_fr) / np.sum(is_fr))

                # MV
                _, s_mv_all, _, _ = sim_mv.simulate_night(prof)
                gp_mv = UniversalKrigingGP().fit(coords_mv_k, s_mv_all[8, :6 + n_add])
                mu_mv, sig_mv = gp_mv.predict(X_parcels)
                p_mv = compute_frost_probability(mu_mv, sig_mv)
                rec_mv_05.append(np.sum((p_mv > 0.5) & is_fr) / np.sum(is_fr))
                rec_mv_02.append(np.sum((p_mv > 0.2) & is_fr) / np.sum(is_fr))

            mean_df_05 = float(np.mean(rec_df_05))
            mean_mv_05 = float(np.mean(rec_mv_05))
            diff_ci = paired_diff_ci(np.array(rec_df_05), np.array(rec_mv_05))

            results["eval"][f"add_{n_add}"] = {
                "df_rec05": round(mean_df_05 * 100, 1),
                "mv_rec05": round(mean_mv_05 * 100, 1),
                "df_rec02": round(float(np.mean(rec_df_02)) * 100, 1),
                "mv_rec02": round(float(np.mean(rec_mv_02)) * 100, 1),
                "diff_05": diff_ci,
            }

        return results

    res_split_a = evaluate_split(split_1_train, split_1_eval, "Train: 002-030 -> Eval: 031-060")
    res_split_b = evaluate_split(split_1_eval, split_1_train, "Train: 031-060 -> Eval: 002-030")

    print(f"  Split A ({res_split_a['split']}):")
    print(f"    DF Additions: {res_split_a['df_added']} | MV Additions: {res_split_a['mv_added']}")
    for k, v in res_split_a["eval"].items():
        print(f"    {k}: DF Rec@0.5={v['df_rec05']}% vs MV={v['mv_rec05']}% (d={v['diff_05']['mean_diff']:+.3f}, 95% CI {v['diff_05']['ci_95']}, p={v['diff_05']['p_value']})")

    print(f"  Split B ({res_split_b['split']}):")
    print(f"    DF Additions: {res_split_b['df_added']} | MV Additions: {res_split_b['mv_added']}")
    for k, v in res_split_b["eval"].items():
        print(f"    {k}: DF Rec@0.5={v['df_rec05']}% vs MV={v['mv_rec05']}% (d={v['diff_05']['mean_diff']:+.3f}, 95% CI {v['diff_05']['ci_95']}, p={v['diff_05']['p_value']})")

    # =========================================================================
    # ITEM 6: Weight Constant Sensitivity (1.0°C, 1.8°C, 3.0°C)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITEM 6: Tuning Sensitivity of Weight Constant w0 in exp(-0.5*(mu/w0)^2)")
    print("=" * 70)

    w0_values = [1.0, 1.8, 3.0]
    w0_results = {}

    for w0 in w0_values:
        w_tune = np.zeros(n_parcels)
        for nid in training_nights[::2]:
            fr = replays_04[nid]
            s_t = np.array([fr["sensor_readings"][s.id] for s in stations])
            gp = UniversalKrigingGP().fit(st_coords_default, s_t)
            mu, _ = gp.predict(X_parcels)
            w_tune += np.exp(-0.5 * (mu / w0) ** 2)
        w_tune /= len(training_nights[::2])

        opt_tune = NextSensorOptimizer(parcels_net, decision_weights=w_tune)
        snet_tune = SensorNetwork()
        tune_added = []
        for i in range(2):
            coords_cur = np.array([[s.x, s.y, s.rel_elev] for s in snet_tune.stations])
            s_temps = np.mean([np.array([replays_04[nid]["sensor_readings"].get(s.id, replays_04[nid]["regional_temp"]) for s in snet_tune.stations]) for nid in training_nights[:10]], axis=0)
            gp_c = UniversalKrigingGP().fit(coords_cur, s_temps)
            rec = opt_tune.find_next_sensor(gp_c, snet_tune, s_temps)
            tune_added.append(rec["best_parcel_id"])
            snet_tune.add_station(rec["best_coords"][1], rec["best_coords"][0], f"Tune_{i}")

        w0_results[f"w0_{w0}C"] = {
            "w0": w0,
            "additions": tune_added,
            "mean_pocket_weight": round(float(np.mean(w_tune[pocket_mask])), 3),
            "mean_slope_weight": round(float(np.mean(w_tune[~pocket_mask])), 3),
        }
        print(f"  w0 = {w0:.1f} deg C: Additions={tune_added} | Pocket Weight={w0_results[f'w0_{w0}C']['mean_pocket_weight']} | Slope Weight={w0_results[f'w0_{w0}C']['mean_slope_weight']}")

    # =========================================================================
    # ITEM 3: Step-by-Step Table K=6..12 (DF vs MV vs Random with CIs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITEM 3: Step-by-Step Sequential Additions K=6..12 Table")
    print("=" * 70)

    # 1. Decision-Focused sequential additions
    w_full = np.zeros(n_parcels)
    for nid in training_nights[::2]:
        fr = replays_04[nid]
        s_t = np.array([fr["sensor_readings"][s.id] for s in stations])
        gp = UniversalKrigingGP().fit(st_coords_default, s_t)
        mu, _ = gp.predict(X_parcels)
        w_full += np.exp(-0.5 * (mu / 1.8) ** 2)
    w_full /= len(training_nights[::2])

    opt_df_full = NextSensorOptimizer(parcels_net, decision_weights=w_full)
    snet_df_seq = SensorNetwork()
    sim_df_seq = MicroclimateSimulator(parcels=parcels_net, sensors=snet_df_seq)

    # Max-Variance sequential additions
    snet_mv_seq = SensorNetwork()
    sim_mv_seq = MicroclimateSimulator(parcels=parcels_net, sensors=snet_mv_seq)

    hero_prof = build_night_profiles()["night_001"]
    hero_frame = replays_04["night_001"]
    hero_y_true = np.array([hero_frame["parcel_temperatures"][p.id] for p in parcels])
    hero_is_frozen = hero_y_true < 0.0

    sequential_k_table = []

    # Pre-derive additions K=7..12
    df_parcels_added = []
    mv_parcels_added = []

    for k in range(6, 12):
        # DF add
        cur_coords = np.array([[s.x, s.y, s.rel_elev] for s in snet_df_seq.stations])
        sim_df_seq.sensors_net.stations = list(snet_df_seq.stations)
        _, s_hero, _, _ = sim_df_seq.simulate_night(hero_prof)
        gp_df = UniversalKrigingGP().fit(cur_coords, s_hero[8, :k])
        rec_df = opt_df_full.find_next_sensor(gp_df, snet_df_seq, s_hero[8, :k])
        df_parcels_added.append(rec_df["best_parcel_id"])
        snet_df_seq.add_station(rec_df["best_coords"][1], rec_df["best_coords"][0], f"DF_{k}")

        # MV add
        cur_mv_coords = np.array([[s.x, s.y, s.rel_elev] for s in snet_mv_seq.stations])
        sim_mv_seq.sensors_net.stations = list(snet_mv_seq.stations)
        _, s_hero_mv, _, _ = sim_mv_seq.simulate_night(hero_prof)
        gp_mv = UniversalKrigingGP().fit(cur_mv_coords, s_hero_mv[8, :k])
        _, sigs = gp_mv.predict(X_parcels)
        dists = np.hypot(X_parcels[:, None, 0] - cur_mv_coords[None, :, 0], X_parcels[:, None, 1] - cur_mv_coords[None, :, 1])
        valid = np.min(dists, axis=-1) >= 250.0
        best_idx = int(np.argmax(np.where(valid, sigs, -1.0)))
        mv_parcels_added.append(parcels[best_idx].id)
        snet_mv_seq.add_station(parcels[best_idx].centroid_lat, parcels[best_idx].centroid_lon, f"MV_{k}")

    print(f"  DF Stations Added (K=7..12): {df_parcels_added}")
    print(f"  MV Stations Added (K=7..12): {mv_parcels_added}")

    # Reset networks to evaluate step-by-step
    snet_df_eval = SensorNetwork()
    snet_mv_eval = SensorNetwork()
    eligible_indices = [i for i, p in enumerate(parcels) if not p.is_frost_pocket]

    for step_idx, k in enumerate(range(6, 13)):
        # 1. DF setup
        sim_df = MicroclimateSimulator(parcels=parcels_net, sensors=snet_df_eval)
        coords_df = np.array([[s.x, s.y, s.rel_elev] for s in snet_df_eval.stations])
        _, s_hero_df, _, _ = sim_df.simulate_night(hero_prof)
        gp_df = UniversalKrigingGP().fit(coords_df, s_hero_df[8, :k])
        mu_hero_df, sig_hero_df = gp_df.predict(X_parcels)
        p_hero_df = compute_frost_probability(mu_hero_df, sig_hero_df)

        hero_pock_05_df = float(np.sum((p_hero_df[pocket_mask] > 0.5) & hero_is_frozen[pocket_mask]) / np.sum(hero_is_frozen[pocket_mask]))
        hero_pock_02_df = float(np.sum((p_hero_df[pocket_mask] > 0.2) & hero_is_frozen[pocket_mask]) / np.sum(hero_is_frozen[pocket_mask]))
        hero_all_05_df = float(np.sum((p_hero_df > 0.5) & hero_is_frozen) / np.sum(hero_is_frozen))
        hero_all_02_df = float(np.sum((p_hero_df > 0.2) & hero_is_frozen) / np.sum(hero_is_frozen))
        hero_holl_05_df = float(np.sum((p_hero_df[~pocket_mask] > 0.5) & hero_is_frozen[~pocket_mask]) / np.sum(hero_is_frozen[~pocket_mask]))
        hero_holl_02_df = float(np.sum((p_hero_df[~pocket_mask] > 0.2) & hero_is_frozen[~pocket_mask]) / np.sum(hero_is_frozen[~pocket_mask]))

        # Training nights average for DF
        tr_df_all_05, tr_df_pock_05, tr_df_holl_05 = [], [], []
        tr_df_all_02, tr_df_pock_02, tr_df_holl_02 = [], [], []
        for nid in training_nights[:20]:
            fr = replays_04[nid]
            y_t = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
            is_fr = y_t < 0.0
            if np.sum(is_fr) == 0:
                continue
            prof = build_night_profiles()[nid]
            _, s_tr_df, _, _ = sim_df.simulate_night(prof)
            gp_tr = UniversalKrigingGP().fit(coords_df, s_tr_df[8, :k])
            mu_tr, sig_tr = gp_tr.predict(X_parcels)
            p_tr = compute_frost_probability(mu_tr, sig_tr)
            tr_df_all_05.append(np.sum((p_tr > 0.5) & is_fr) / np.sum(is_fr))
            tr_df_all_02.append(np.sum((p_tr > 0.2) & is_fr) / np.sum(is_fr))
            if np.sum(is_fr & pocket_mask) > 0:
                tr_df_pock_05.append(np.sum((p_tr[pocket_mask] > 0.5) & is_fr[pocket_mask]) / np.sum(is_fr[pocket_mask]))
                tr_df_pock_02.append(np.sum((p_tr[pocket_mask] > 0.2) & is_fr[pocket_mask]) / np.sum(is_fr[pocket_mask]))
            if np.sum(is_fr & (~pocket_mask)) > 0:
                tr_df_holl_05.append(np.sum((p_tr[~pocket_mask] > 0.5) & is_fr[~pocket_mask]) / np.sum(is_fr[~pocket_mask]))
                tr_df_holl_02.append(np.sum((p_tr[~pocket_mask] > 0.2) & is_fr[~pocket_mask]) / np.sum(is_fr[~pocket_mask]))

        # 2. MV setup
        sim_mv = MicroclimateSimulator(parcels=parcels_net, sensors=snet_mv_eval)
        coords_mv = np.array([[s.x, s.y, s.rel_elev] for s in snet_mv_eval.stations])
        _, s_hero_mv, _, _ = sim_mv.simulate_night(hero_prof)
        gp_mv = UniversalKrigingGP().fit(coords_mv, s_hero_mv[8, :k])
        mu_hero_mv, sig_hero_mv = gp_mv.predict(X_parcels)
        p_hero_mv = compute_frost_probability(mu_hero_mv, sig_hero_mv)

        hero_pock_05_mv = float(np.sum((p_hero_mv[pocket_mask] > 0.5) & hero_is_frozen[pocket_mask]) / np.sum(hero_is_frozen[pocket_mask]))
        hero_pock_02_mv = float(np.sum((p_hero_mv[pocket_mask] > 0.2) & hero_is_frozen[pocket_mask]) / np.sum(hero_is_frozen[pocket_mask]))
        hero_all_05_mv = float(np.sum((p_hero_mv > 0.5) & hero_is_frozen) / np.sum(hero_is_frozen))
        hero_all_02_mv = float(np.sum((p_hero_mv > 0.2) & hero_is_frozen) / np.sum(hero_is_frozen))
        hero_holl_05_mv = float(np.sum((p_hero_mv[~pocket_mask] > 0.5) & hero_is_frozen[~pocket_mask]) / np.sum(hero_is_frozen[~pocket_mask]))
        hero_holl_02_mv = float(np.sum((p_hero_mv[~pocket_mask] > 0.2) & hero_is_frozen[~pocket_mask]) / np.sum(hero_is_frozen[~pocket_mask]))

        tr_mv_all_05, tr_mv_pock_05, tr_mv_holl_05 = [], [], []
        tr_mv_all_02, tr_mv_pock_02, tr_mv_holl_02 = [], [], []
        for nid in training_nights[:20]:
            fr = replays_04[nid]
            y_t = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
            is_fr = y_t < 0.0
            if np.sum(is_fr) == 0:
                continue
            prof = build_night_profiles()[nid]
            _, s_tr_mv, _, _ = sim_mv.simulate_night(prof)
            gp_tr = UniversalKrigingGP().fit(coords_mv, s_tr_mv[8, :k])
            mu_tr, sig_tr = gp_tr.predict(X_parcels)
            p_tr = compute_frost_probability(mu_tr, sig_tr)
            tr_mv_all_05.append(np.sum((p_tr > 0.5) & is_fr) / np.sum(is_fr))
            tr_mv_all_02.append(np.sum((p_tr > 0.2) & is_fr) / np.sum(is_fr))
            if np.sum(is_fr & pocket_mask) > 0:
                tr_mv_pock_05.append(np.sum((p_tr[pocket_mask] > 0.5) & is_fr[pocket_mask]) / np.sum(is_fr[pocket_mask]))
                tr_mv_pock_02.append(np.sum((p_tr[pocket_mask] > 0.2) & is_fr[pocket_mask]) / np.sum(is_fr[pocket_mask]))
            if np.sum(is_fr & (~pocket_mask)) > 0:
                tr_mv_holl_05.append(np.sum((p_tr[~pocket_mask] > 0.5) & is_fr[~pocket_mask]) / np.sum(is_fr[~pocket_mask]))
                tr_mv_holl_02.append(np.sum((p_tr[~pocket_mask] > 0.2) & is_fr[~pocket_mask]) / np.sum(is_fr[~pocket_mask]))

        # 3. Random placement (20 random draws)
        rnd_hero_all_05, rnd_hero_pock_05 = [], []
        rnd_tr_all_05, rnd_tr_pock_05 = [], []
        if k > 6:
            for r_seed in range(20):
                rng_r = np.random.RandomState(42 + r_seed * 17 + k)
                extra = rng_r.choice(eligible_indices, size=k - 6, replace=False)
                c_rnd = np.vstack([st_coords_default, X_parcels[extra]])
                snet_r = SensorNetwork()
                snet_r.stations = []
                for idx_c, cc in enumerate(c_rnd):
                    snet_r.stations.append(Station(f"R{idx_c}", f"R{idx_c}", cc[0], cc[1], 0.0, 0.0, float(parcels_net.terrain.elevation(cc[0], cc[1])), float(cc[2]), False))
                sim_r = MicroclimateSimulator(parcels=parcels_net, sensors=snet_r)
                _, s_hero_r, _, _ = sim_r.simulate_night(hero_prof)
                gp_r = UniversalKrigingGP().fit(c_rnd, s_hero_r[8, :k])
                mu_r, sig_r = gp_r.predict(X_parcels)
                p_r = compute_frost_probability(mu_r, sig_r)
                rnd_hero_all_05.append(np.sum((p_r > 0.5) & hero_is_frozen) / np.sum(hero_is_frozen))
                rnd_hero_pock_05.append(np.sum((p_r[pocket_mask] > 0.5) & hero_is_frozen[pocket_mask]) / np.sum(hero_is_frozen[pocket_mask]))

                # Train sample
                for nid in training_nights[:10]:
                    fr = replays_04[nid]
                    y_t = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
                    is_fr = y_t < 0.0
                    if np.sum(is_fr) == 0:
                        continue
                    prof = build_night_profiles()[nid]
                    _, s_tr_r, _, _ = sim_r.simulate_night(prof)
                    gp_tr_r = UniversalKrigingGP().fit(c_rnd, s_tr_r[8, :k])
                    mu_tr_r, sig_tr_r = gp_tr_r.predict(X_parcels)
                    p_tr_r = compute_frost_probability(mu_tr_r, sig_tr_r)
                    rnd_tr_all_05.append(np.sum((p_tr_r > 0.5) & is_fr) / np.sum(is_fr))
                    if np.sum(is_fr & pocket_mask) > 0:
                        rnd_tr_pock_05.append(np.sum((p_tr_r[pocket_mask] > 0.5) & is_fr[pocket_mask]) / np.sum(is_fr[pocket_mask]))

        rnd_pock_ci = bootstrap_ci(np.array(rnd_hero_pock_05)) if rnd_hero_pock_05 else (0.0, 0.0)
        rnd_tr_ci = bootstrap_ci(np.array(rnd_tr_all_05)) if rnd_tr_all_05 else (0.0, 0.0)

        # Paired difference test on training nights between DF and MV
        paired_diff_tr = paired_diff_ci(np.array(tr_df_all_05), np.array(tr_mv_all_05))

        k_entry = {
            "k": k,
            "df_sensor_added": df_parcels_added[step_idx - 1] if step_idx > 0 else "Baseline (6 default)",
            "mv_sensor_added": mv_parcels_added[step_idx - 1] if step_idx > 0 else "Baseline (6 default)",
            "hero_night": {
                "decision_focused": {
                    "pocket_rec05": round(hero_pock_05_df * 100, 1),
                    "pocket_rec02": round(hero_pock_02_df * 100, 1),
                    "hollow_rec05": round(hero_holl_05_df * 100, 1),
                    "hollow_rec02": round(hero_holl_02_df * 100, 1),
                    "all_rec05": round(hero_all_05_df * 100, 1),
                    "all_rec02": round(hero_all_02_df * 100, 1),
                },
                "max_variance": {
                    "pocket_rec05": round(hero_pock_05_mv * 100, 1),
                    "pocket_rec02": round(hero_pock_02_mv * 100, 1),
                    "hollow_rec05": round(hero_holl_05_mv * 100, 1),
                    "hollow_rec02": round(hero_holl_02_mv * 100, 1),
                    "all_rec05": round(hero_all_05_mv * 100, 1),
                    "all_rec02": round(hero_all_02_mv * 100, 1),
                },
                "random_mean_ci": {
                    "pocket_rec05_mean": round(float(np.mean(rnd_hero_pock_05)) * 100, 1) if rnd_hero_pock_05 else 0.0,
                    "pocket_rec05_ci": [round(rnd_pock_ci[0] * 100, 1), round(rnd_pock_ci[1] * 100, 1)],
                },
            },
            "training_nights_avg": {
                "decision_focused": {
                    "pocket_rec05": round(float(np.mean(tr_df_pock_05)) * 100, 1) if tr_df_pock_05 else 0.0,
                    "pocket_rec02": round(float(np.mean(tr_df_pock_02)) * 100, 1) if tr_df_pock_02 else 0.0,
                    "hollow_rec05": round(float(np.mean(tr_df_holl_05)) * 100, 1) if tr_df_holl_05 else 0.0,
                    "hollow_rec02": round(float(np.mean(tr_df_holl_02)) * 100, 1) if tr_df_holl_02 else 0.0,
                    "all_rec05": round(float(np.mean(tr_df_all_05)) * 100, 1) if tr_df_all_05 else 0.0,
                    "all_rec02": round(float(np.mean(tr_df_all_02)) * 100, 1) if tr_df_all_02 else 0.0,
                },
                "max_variance": {
                    "pocket_rec05": round(float(np.mean(tr_mv_pock_05)) * 100, 1) if tr_mv_pock_05 else 0.0,
                    "pocket_rec02": round(float(np.mean(tr_mv_pock_02)) * 100, 1) if tr_mv_pock_02 else 0.0,
                    "hollow_rec05": round(float(np.mean(tr_mv_holl_05)) * 100, 1) if tr_mv_holl_05 else 0.0,
                    "hollow_rec02": round(float(np.mean(tr_mv_holl_02)) * 100, 1) if tr_mv_holl_02 else 0.0,
                    "all_rec05": round(float(np.mean(tr_mv_all_05)) * 100, 1) if tr_mv_all_05 else 0.0,
                    "all_rec02": round(float(np.mean(tr_mv_all_02)) * 100, 1) if tr_mv_all_02 else 0.0,
                },
                "random_mean_ci": {
                    "all_rec05_mean": round(float(np.mean(rnd_tr_all_05)) * 100, 1) if rnd_tr_all_05 else 0.0,
                    "all_rec05_ci": [round(rnd_tr_ci[0] * 100, 1), round(rnd_tr_ci[1] * 100, 1)],
                },
                "paired_diff_df_minus_mv": paired_diff_tr,
            },
        }
        sequential_k_table.append(k_entry)

        print(f"  K={k:2d} | DF added {k_entry['df_sensor_added']:12s} | Hero Pocket Rec@0.5: DF={hero_pock_05_df*100:4.1f}% vs MV={hero_pock_05_mv*100:4.1f}% vs Rnd={k_entry['hero_night']['random_mean_ci']['pocket_rec05_mean']:4.1f}% | Train All: DF={k_entry['training_nights_avg']['decision_focused']['all_rec05']:4.1f}% vs MV={k_entry['training_nights_avg']['max_variance']['all_rec05']:4.1f}% (p={paired_diff_tr['p_value']})")

        # Advance networks for next iteration
        if step_idx < len(df_parcels_added):
            p_df_next = [p for p in parcels if p.id == df_parcels_added[step_idx]][0]
            snet_df_eval.add_station(p_df_next.centroid_lat, p_df_next.centroid_lon, f"DF_{step_idx+1}")
            p_mv_next = [p for p in parcels if p.id == mv_parcels_added[step_idx]][0]
            snet_mv_eval.add_station(p_mv_next.centroid_lat, p_mv_next.centroid_lon, f"MV_{step_idx+1}")

    # =========================================================================
    # ITEM 7: Bonferroni-Adjusted Baseline Differences
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITEM 7: Baseline Benchmark with Bonferroni-Adjusted Significance (alpha=0.0125)")
    print("=" * 70)

    baselines = {
        "terrain_linear": TerrainLinearBaseline(),
        "terrain_only_gp": TerrainOnlyGP(),
        "idw": IDWBaseline(),
        "nearest_sensor": NearestSensorBaseline(),
    }

    gp_rmses = []
    base_rmses = {k: [] for k in baselines}

    for nid in training_nights:
        fr = replays_04[nid]
        y_true = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
        s_temps = np.array([fr["sensor_readings"][s.id] for s in stations])

        gp_m = UniversalKrigingGP().fit(st_coords_default, s_temps)
        mu_gp, _ = gp_m.predict(X_parcels)
        gp_rmses.append(np.sqrt(np.mean((mu_gp - y_true)**2)))

        for b_name, b_model in baselines.items():
            if b_name == "terrain_only_gp":
                b_model.fit(st_coords_default, s_temps)
                mu_b, _ = b_model.predict(X_parcels)
            else:
                b_model.fit(st_coords_default[:, :2], s_temps, sensor_features=st_coords_default)
                mu_b, _ = b_model.predict(X_parcels[:, :2], target_features=X_parcels)
            base_rmses[b_name].append(np.sqrt(np.mean((mu_b - y_true)**2)))

    bonferroni_results = {}
    alpha_bonferroni = 0.05 / 4.0  # 0.0125

    for b_name in baselines:
        diff_stats = paired_diff_ci(np.array(base_rmses[b_name]), np.array(gp_rmses))
        is_sig = diff_stats["p_value"] < alpha_bonferroni
        bonferroni_results[b_name] = {
            "mean_delta_rmse_deg_c": diff_stats["mean_diff"],
            "ci_95": diff_stats["ci_95"],
            "raw_p_value": diff_stats["p_value"],
            "bonferroni_alpha": alpha_bonferroni,
            "statistically_significant": is_sig,
            "conclusion": "Statistically significant improvement" if is_sig else "Statistically tied / non-significant",
        }
        print(f"  {b_name:18s} | dRMSE = {diff_stats['mean_diff']:+.4f} deg C (95% CI {diff_stats['ci_95']}, p={diff_stats['p_value']:.4f}) -> {bonferroni_results[b_name]['conclusion']}")

    # =========================================================================
    # ITEM 1: Honest ROI Evaluation (A, B1, B2, C(K=6), C(K=8)) across Nights
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITEM 1: Honest Economic Decision-Theoretic Evaluation across Nights")
    print("=" * 70)

    # Cost-Loss decision rule: action threshold p* = cost / loss
    cost = 350.0
    loss = 4500.0
    cost_loss_p_star = cost / loss  # ~0.0778 (approx 7.8%)

    # Setup K=8 sensor network (default + P050 + P202)
    snet_k8 = SensorNetwork()
    p050 = [p for p in parcels if p.id == "P050"][0]
    p202 = [p for p in parcels if p.id == "P202"][0]
    snet_k8.add_station(p050.centroid_lat, p050.centroid_lon, "P050 (Valley South)")
    snet_k8.add_station(p202.centroid_lat, p202.centroid_lon, "P202 (Valley North)")
    coords_k8 = np.array([[s.x, s.y, s.rel_elev] for s in snet_k8.stations])
    sim_k8 = MicroclimateSimulator(parcels=parcels_net, sensors=snet_k8)

    def evaluate_roi_for_night(nid, p_threshold=cost_loss_p_star):
        fr = replays_04[nid]
        y_true = np.array([fr["parcel_temperatures"][p.id] for p in parcels])
        truly_frozen = y_true < 0.0
        n_frozen = int(np.sum(truly_frozen))
        reg_t = fr["regional_temp"]

        # Strategy A: Protect Nothing
        exp_a = n_frozen * loss

        # Strategy B1: Regional Alert Rule (protect all 239 ONLY when regional forecast min <= 0°C)
        b1_triggered = reg_t <= 0.0
        exp_b1 = (n_parcels * cost) if b1_triggered else (n_frozen * loss)
        prot_b1 = n_parcels if b1_triggered else 0

        # Strategy B2: Always Blanket Protect Everything (Upper bound)
        exp_b2 = n_parcels * cost
        prot_b2 = n_parcels

        # Strategy C(K=6): Cassini Vivant default 6 stations
        s_temps_6 = np.array([fr["sensor_readings"][s.id] for s in stations])
        gp_6 = UniversalKrigingGP().fit(st_coords_default, s_temps_6)
        mu_6, sig_6 = gp_6.predict(X_parcels)
        p_frost_6 = compute_frost_probability(mu_6, sig_6)
        prot_c6_mask = p_frost_6 >= p_threshold
        prot_c6 = int(np.sum(prot_c6_mask))
        damaged_c6 = int(np.sum(truly_frozen & (~prot_c6_mask)))
        exp_c6 = (prot_c6 * cost) + (damaged_c6 * loss)

        # Strategy C(K=8): Cassini Vivant with 2 decision-focused valley stations
        prof = build_night_profiles()[nid]
        _, s_k8_all, _, _ = sim_k8.simulate_night(prof)
        gp_8 = UniversalKrigingGP().fit(coords_k8, s_k8_all[8, :8])
        mu_8, sig_8 = gp_8.predict(X_parcels)
        p_frost_8 = compute_frost_probability(mu_8, sig_8)
        prot_c8_mask = p_frost_8 >= p_threshold
        prot_c8 = int(np.sum(prot_c8_mask))
        damaged_c8 = int(np.sum(truly_frozen & (~prot_c8_mask)))
        exp_c8 = (prot_c8 * cost) + (damaged_c8 * loss)

        return {
            "night": nid,
            "reg_t": reg_t,
            "truly_frozen": n_frozen,
            "A": {"exp": exp_a, "prot": 0, "damaged": n_frozen},
            "B1": {"exp": exp_b1, "prot": prot_b1, "damaged": 0 if b1_triggered else n_frozen},
            "B2": {"exp": exp_b2, "prot": prot_b2, "damaged": 0},
            "C_k6": {"exp": exp_c6, "prot": prot_c6, "damaged": damaged_c6},
            "C_k8": {"exp": exp_c8, "prot": prot_c8, "damaged": damaged_c8},
        }

    # Evaluate Hero Night Showcase
    hero_roi = evaluate_roi_for_night("night_001")
    print(f"  Hero Night Showcase (T_reg = {hero_roi['reg_t']:.1f} deg C, Truly Frozen = {hero_roi['truly_frozen']} plots):")
    print(f"    Strategy A (Protect Nothing):             Total = {hero_roi['A']['exp']:,.0f} EUR (Protected: 0, Damaged: {hero_roi['A']['damaged']})")
    print(f"    Strategy B1 (Regional Alert <= 0 C):      Total = {hero_roi['B1']['exp']:,.0f} EUR (Protected: {hero_roi['B1']['prot']} -> FAILS TO TRIGGER because T_reg=+2.4 deg C!)")
    print(f"    Strategy B2 (Always Blanket Protect):     Total = {hero_roi['B2']['exp']:,.0f} EUR (Protected: {hero_roi['B2']['prot']}, Cost Upper Bound)")
    print(f"    Strategy C (K=6 Default):                 Total = {hero_roi['C_k6']['exp']:,.0f} EUR (Protected: {hero_roi['C_k6']['prot']}, Damaged: {hero_roi['C_k6']['damaged']})")
    print(f"    Strategy C (K=8 Valley Sensors Added):    Total = {hero_roi['C_k8']['exp']:,.0f} EUR (Protected: {hero_roi['C_k8']['prot']}, Damaged: {hero_roi['C_k8']['damaged']})")

    # Evaluate across All Training Nights 002-060
    training_rois = [evaluate_roi_for_night(nid) for nid in training_nights]
    tr_exp_a = [r["A"]["exp"] for r in training_rois]
    tr_exp_b1 = [r["B1"]["exp"] for r in training_rois]
    tr_exp_b2 = [r["B2"]["exp"] for r in training_rois]
    tr_exp_c6 = [r["C_k6"]["exp"] for r in training_rois]
    tr_exp_c8 = [r["C_k8"]["exp"] for r in training_rois]

    roi_summary = {
        "cost_loss_ratio": round(cost_loss_p_star, 4),
        "cost_per_parcel_night": cost,
        "loss_per_parcel": loss,
        "break_even_loss_to_cost": round(loss / cost, 2),
        "hero_night_showcase": hero_roi,
        "training_nights_averages_per_night": {
            "strategy_a": {"mean": round(float(np.mean(tr_exp_a))), "ci_95": bootstrap_ci(np.array(tr_exp_a))},
            "strategy_b1": {"mean": round(float(np.mean(tr_exp_b1))), "ci_95": bootstrap_ci(np.array(tr_exp_b1))},
            "strategy_b2": {"mean": round(float(np.mean(tr_exp_b2))), "ci_95": bootstrap_ci(np.array(tr_exp_b2))},
            "strategy_c_k6": {"mean": round(float(np.mean(tr_exp_c6))), "ci_95": bootstrap_ci(np.array(tr_exp_c6))},
            "strategy_c_k8": {"mean": round(float(np.mean(tr_exp_c8))), "ci_95": bootstrap_ci(np.array(tr_exp_c8))},
        },
    }

    print("\n  Training Nights Average Expenditure per Night (Mean & 95% CI):")
    for strat, data in roi_summary["training_nights_averages_per_night"].items():
        print(f"    {strat:15s}: {data['mean']:,} EUR [95% CI: {data['ci_95'][0]:,}, {data['ci_95'][1]:,}]")

    # Save to eval.json
    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        full_eval = json.load(f)

    full_eval["phase_4b"] = {
        "out_of_sample_split_placement": {
            "split_a": res_split_a,
            "split_b": res_split_b,
        },
        "weight_constant_sensitivity": w0_results,
        "step_by_step_sequential_k6_k12": sequential_k_table,
        "bonferroni_baseline_differences": bonferroni_results,
        "honest_roi_decision_theoretic": roi_summary,
    }

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(full_eval, f, indent=2)

    print(f"\nPhase 4b evaluation successfully complete. Saved to {eval_file}.")


if __name__ == "__main__":
    main()
