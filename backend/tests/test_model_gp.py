"""Unit tests for Universal Kriging GP, baselines, mesh, and next-best sensor placement."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.model.baselines import (
    IDWBaseline,
    NearestSensorBaseline,
    TerrainLinearBaseline,
    TerrainOnlyGP,
)
from backend.model.dem_features import DEMFeatureExtractor
from backend.model.frost import classify_risk_tier, compute_frost_probability
from backend.model.gp import UniversalKrigingGP
from backend.model.mesh import compute_delaunay_mesh
from backend.model.next_sensor import NextSensorOptimizer
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork

client = TestClient(app)


@pytest.fixture
def setup_models():
    parcels_net = ParcelNetwork()
    sensors_net = SensorNetwork()
    dem_extractor = DEMFeatureExtractor(parcels_net.terrain, parcels_net.parcels)
    stations = sensors_net.stations

    # Build [x, y, rel_elev] coordinate arrays
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in stations])
    parcel_coords = np.array([
        [p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels_net.parcels
    ])

    # Sample sensor readings (typical mild frost night: all sensors above 0°C)
    dummy_temps = np.array([1.8, 0.4, 3.1, 2.1, 2.3, 3.1])

    gp = UniversalKrigingGP()
    gp.fit(st_coords, dummy_temps)

    return parcels_net, sensors_net, dem_extractor, gp, st_coords, parcel_coords, dummy_temps


def test_gp_sigma_positive_everywhere(setup_models):
    """Verify predictive uncertainty sigma >= 0 across all parcels."""
    _, _, _, gp, _, parcel_coords, _ = setup_models
    mu, sigma = gp.predict(parcel_coords)

    assert len(mu) == len(parcel_coords)
    assert len(sigma) == len(parcel_coords)
    assert np.all(sigma >= 0.0), f"Found negative sigma: {sigma.min()}"
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))


def test_gp_sigma_lower_near_sensors(setup_models):
    """Verify uncertainty sigma is lower at parcels adjacent to stations than at remote parcels."""
    parcels_net, sensors_net, _, gp, _, parcel_coords, _ = setup_models
    _, sigma = gp.predict(parcel_coords)

    st_positions = np.array([[s.x, s.y] for s in sensors_net.stations])
    pc_xy = parcel_coords[:, :2]

    dists = np.min(
        np.hypot(
            pc_xy[:, None, 0] - st_positions[None, :, 0],
            pc_xy[:, None, 1] - st_positions[None, :, 1],
        ),
        axis=-1,
    )

    near_mask = dists < 300.0
    far_mask = dists > 1400.0

    sigma_near = float(np.mean(sigma[near_mask]))
    sigma_far = float(np.mean(sigma[far_mask]))

    assert sigma_near < sigma_far, (
        f"Expected sigma near ({sigma_near:.3f}) < sigma far ({sigma_far:.3f})"
    )


def test_variance_monotonicity_with_sensor_additions(setup_models):
    """Phase 3b requirement: with frozen hyperparameters, total posterior variance is
    monotonically non-increasing as sensors are added (K=6..12).
    """
    parcels_net, sensors_net, _, _, st_coords, parcel_coords, dummy_temps = setup_models

    prev_mean_sigma = float("inf")
    current_sensor_coords = st_coords.copy()
    current_temps = dummy_temps.copy()

    for k in range(6, 13):
        gp = UniversalKrigingGP()
        gp.fit(current_sensor_coords, current_temps)
        _, sigma = gp.predict(parcel_coords)
        mean_sigma = float(np.mean(sigma))

        assert mean_sigma <= prev_mean_sigma + 1e-9, (
            f"Variance monotonicity violated: K={k} sigma_bar={mean_sigma:.4f} > "
            f"K={k-1} sigma_bar={prev_mean_sigma:.4f}"
        )
        prev_mean_sigma = mean_sigma

        if k < 12:
            # Add a dummy sensor at the centroid of a random far-away parcel
            new_idx = (k - 6) * 30 + 10  # Spread across parcels: P010, P040, P070, ...
            new_coord = parcel_coords[min(new_idx, len(parcel_coords) - 1)]
            current_sensor_coords = np.vstack([current_sensor_coords, new_coord])
            current_temps = np.append(current_temps, float(np.mean(current_temps)))


def test_decision_focused_rule_outperforms_max_variance():
    """Phase 3c requirement: non-circular validation test confirming decision-focused sensor placement
    achieves higher mean frozen-parcel recall than the unweighted max-variance rule on training nights 002-060.
    """
    import json
    from backend.app.config import DATA_DIR
    eval_file = DATA_DIR / "eval.json"
    assert eval_file.exists(), "eval.json must exist"

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    p3c = data.get("phase_3c", {}).get("decision_focused_vs_rules_training_nights", {})
    assert "decision_focused" in p3c
    assert "max_variance" in p3c
    assert "random" in p3c

    # Check Add +1, +2, +3
    for add_k in ["add_1", "add_2", "add_3"]:
        df_rec = p3c["decision_focused"][add_k]["rec05"]
        mv_rec = p3c["max_variance"][add_k]["rec05"]
        rnd_rec = p3c["random"][add_k]["rec05"]

        assert df_rec >= mv_rec, (
            f"Expected decision-focused recall ({df_rec:.3f}) >= max-variance ({mv_rec:.3f}) at {add_k}"
        )
        assert df_rec > rnd_rec, (
            f"Expected decision-focused recall ({df_rec:.3f}) > random ({rnd_rec:.3f}) at {add_k}"
        )


def test_next_best_sensor_selection(setup_models):
    """Verify next best sensor math: local uncertainty drops, network mean doesn't increase."""
    parcels_net, sensors_net, _, gp, _, parcel_coords, dummy_temps = setup_models
    optimizer = NextSensorOptimizer(parcels_net)

    rec = optimizer.find_next_sensor(gp, sensors_net, dummy_temps)

    assert "best_parcel_id" in rec
    assert rec["best_parcel_id"].startswith("P")
    assert rec["current_sigma"] > rec["expected_new_local_sigma"]  # Local uncertainty drops
    assert rec["sigma_reduction"] >= 0.0
    assert rec["network_mean_sigma_after"] <= rec["network_mean_sigma_current"]


def test_baselines_prediction_contracts(setup_models):
    """Verify all 4 baseline models conform to the prediction contract."""
    parcels_net, sensors_net, dem_extractor, _, st_coords, parcel_coords, dummy_temps = setup_models
    st_positions = st_coords[:, :2]
    target_coords = parcel_coords[:, :2]

    # TerrainOnlyGP takes [x, y, rel_elev] coordinates
    terrain_gp = TerrainOnlyGP()
    terrain_gp.fit(st_coords, dummy_temps)
    mu_tgp, sigma_tgp = terrain_gp.predict(parcel_coords)
    assert len(mu_tgp) == len(parcels_net.parcels)
    assert len(sigma_tgp) == len(parcels_net.parcels)
    assert np.all(sigma_tgp >= 0.0)

    # Spatial baselines
    baselines = [
        NearestSensorBaseline().fit(st_positions, dummy_temps),
        IDWBaseline().fit(st_positions, dummy_temps),
        TerrainLinearBaseline().fit(
            st_positions, dummy_temps,
            sensor_features=dem_extractor.get_parcel_feature_matrix(variant="A")[:6],
        ),
    ]

    for model in baselines:
        mu, sigma = model.predict(
            target_coords,
            target_features=dem_extractor.get_parcel_feature_matrix(variant="A"),
        )
        assert len(mu) == len(parcels_net.parcels)
        assert len(sigma) == len(parcels_net.parcels)
        assert np.all(sigma >= 0.0)


def test_api_predict():
    """Smoke test GET /api/predict."""
    res = client.get("/api/predict?night=night_001&hour=4")
    assert res.status_code == 200
    data = res.json()
    assert data["night"] == "night_001"
    assert data["hour"] == 4
    assert len(data["predictions"]) == 239

    p0 = data["predictions"][0]
    assert "parcel_id" in p0
    assert "mu" in p0
    assert "sigma" in p0
    assert "p_frost" in p0
    assert "is_watch_tier" in p0
    assert "is_critical" in p0
    assert 0.0 <= p0["p_frost"] <= 1.0


def test_api_mesh_and_next_sensor_and_eval():
    """Smoke test /api/mesh, /api/next-sensor, /api/eval."""
    # Mesh
    mesh_res = client.get("/api/mesh")
    assert mesh_res.status_code == 200
    mesh_data = mesh_res.json()
    assert mesh_data["type"] == "FeatureCollection"
    assert mesh_data["metadata"]["station_count"] == 6
    assert len(mesh_data["features"]) > 0

    # Next-sensor
    next_res = client.post("/api/next-sensor?night=night_001&hour=4")
    assert next_res.status_code == 200
    next_data = next_res.json()
    assert next_data["status"] == "success"
    assert "recommendation" in next_data
    assert "best_parcel_id" in next_data["recommendation"]

    # Eval
    eval_res = client.get("/api/eval")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "benchmarks" in eval_data or "item_5_baselines_benchmark" in eval_data
    assert "headline_analysis" in eval_data or "item_6_headline_analysis" in eval_data
    assert "hero_diagnostics" in eval_data or "item_1_hero_diagnostic" in eval_data


def test_frost_probability_and_risk_tiers():
    """Verify frost probability and risk tier classification."""
    mu = np.array([2.0, 0.5, -1.0, -3.0])
    sigma = np.array([0.5, 0.8, 0.6, 1.0])

    p_frost = compute_frost_probability(mu, sigma)
    assert len(p_frost) == 4
    assert np.all((0.0 <= p_frost) & (p_frost <= 1.0))
    # Colder parcels should have higher frost probability
    assert p_frost[3] > p_frost[0]
    assert p_frost[2] > p_frost[0]

    critical, watch = classify_risk_tier(p_frost, mu, sigma)
    assert len(critical) == 4
    assert len(watch) == 4


def test_api_roi():
    """Verify honest ROI calculation contract and cost-loss decision theory."""
    res = client.get("/api/roi?night=night_001&protect_threshold=0.08&cost_per_parcel_night=350&loss_per_parcel=4500")
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "ILLUSTRATIVE — DEMO MODEL"
    assert data["is_hero_showcase"] is True
    assert data["regional_temp"] == 2.4

    params = data["parameters"]
    assert params["break_even_ratio"] == pytest.approx(12.86, rel=1e-2)
    assert params["cost_loss_threshold"] == pytest.approx(0.0778, rel=1e-2)
    assert params["total_parcels"] == 239
    assert params["truly_frozen_parcels"] == 49

    strats = data["strategies"]
    # Strategy A: Protect nothing
    assert strats["strategy_a"]["parcels_protected"] == 0
    assert strats["strategy_a"]["parcels_damaged"] == 49
    assert strats["strategy_a"]["total_expenditure"] == 220500.0

    # Strategy B1: Regional alert rule -> DOES NOT TRIGGER on hero night (+2.4°C)
    assert strats["strategy_b1"]["alert_triggered"] is False
    assert strats["strategy_b1"]["parcels_protected"] == 0
    assert strats["strategy_b1"]["parcels_damaged"] == 49
    assert strats["strategy_b1"]["total_expenditure"] == 220500.0

    # Strategy B2: Blanket protection -> always protects all 239 parcels
    assert strats["strategy_b2"]["parcels_protected"] == 239
    assert strats["strategy_b2"]["parcels_damaged"] == 0
    assert strats["strategy_b2"]["total_expenditure"] == 239 * 350.0  # 83650.0

    # Strategy C (K=8: Default + 2 Valley Sensors) saves significant damage
    assert strats["strategy_c_k8"]["parcels_protected"] > 0
    assert strats["strategy_c_k8"]["total_expenditure"] < strats["strategy_a"]["total_expenditure"]


def test_phase_4c_unified_results_consistency():
    """Verify Phase 4c results consistency: figures come from eval.json and denominators (n) are recorded."""
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    assert eval_file.exists(), "eval.json must exist"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    # 1. Metadata and Provenance
    meta = ev["metadata"]
    assert "git_commit_hash" in meta
    assert len(meta["git_commit_hash"]) == 40
    assert meta["default_pipeline"] == "farm_only_calibration_b"
    prov = meta["provenance"]
    assert "farm_only_b_default" in prov
    assert "dense_survey_a_sensitivity" in prov
    assert "oracle_generator_matched_sensitivity" in prov
    assert prov["oracle_generator_matched_sensitivity"]["role"].startswith("Theoretical upper-bound")

    # 2. Night types classification and catalog totals
    nt = ev["night_types_catalog"]
    assert nt["total_nights"] == 60
    assert nt["ordinary_frost"]["count"] == 3
    assert nt["inversion_frost"]["count"] == 38
    assert nt["no_frost"]["count"] == 19
    assert nt["ordinary_frost"]["count"] + nt["inversion_frost"]["count"] + nt["no_frost"]["count"] == 60
    assert nt["total_frozen_parcel_nights"] == 1892  # 49 hero + 93 H1 + 1750 H2

    # 3. Sigma recalibration: coverage & Brier before and after
    recal = ev["sigma_recalibration"]
    assert recal["active_s_multiplier"] == pytest.approx(1.855, rel=1e-2)
    sp1 = recal["split_1_train_h1_test_h2"]
    assert sp1["test_coverage_after"] > sp1["test_coverage_before"]
    assert sp1["test_brier_after"] < sp1["test_brier_before"]  # Brier improves
    sp2 = recal["split_2_train_h2_test_h1"]
    assert sp2["test_coverage_after"] > sp2["test_coverage_before"]
    assert sp2["test_brier_after"] < sp2["test_brier_before"]  # Brier improves
    pool = recal["pooled_catalog_calibration"]
    assert pool["coverage_before"] == pytest.approx(71.2, abs=0.5)
    assert pool["coverage_after"] == pytest.approx(90.0, abs=0.5)
    assert pool["brier_after"] < pool["brier_before"]

    # 4. Sequential K=6..12 table and recorded denominators
    k_traj = ev["sequential_k6_to_k12_trajectory"]
    assert len(k_traj) == 7  # K=6..12
    for row in k_traj:
        assert "k" in row
        # Hero denominators
        h_data = row["hero_night"]
        assert h_data["pocket"]["n"] == 21
        assert h_data["hollow"]["n"] == 28
        assert h_data["all_frozen"]["n"] == 49
        assert "rec_05" in h_data["all_frozen"]
        assert "rec_02" in h_data["all_frozen"]
        assert "rec_p_star" in h_data["all_frozen"]
        # Training denominators
        tr_data = row["training_nights_averages"]
        assert tr_data["pocket_parcels"]["n_nights"] == 40
        assert tr_data["inversion_nights"]["n_nights"] == 37
        assert tr_data["ordinary_frost_nights"]["n_nights"] == 3
        assert tr_data["all_frost_nights"]["n_nights"] == 40

    # Denominators reconciliation explanation
    recon = ev["denominators_and_reconciliation"]
    assert "multiples_of_one_ninth_explanation" in recon
    assert recon["n_nights_pocket_frost"] == 40

    # 5. Out of sample temporal splits with Random baseline
    splits = ev["out_of_sample_split_placement"]
    assert "split_a_train_002_030_eval_031_060" in splits
    assert "split_b_train_031_060_eval_002_030" in splits
    assert "random_baseline" in splits["split_a_train_002_030_eval_031_060"]
    assert "random_baseline" in splits["split_b_train_031_060_eval_002_030"]
    assert "explanation_of_difference" in splits

    # 6. Disaggregated ROI: false alarms, night types, sensitivity curves
    roi = ev["disaggregated_roi_and_sensitivities"]
    assert "false_alarm_cost" in roi["hero_night_showcase"]["strategy_c_k8"]
    assert "loss_to_cost_sensitivity_curve" in roi
    assert "climatological_frequency_reweighting" in roi
    assert len(roi["loss_to_cost_sensitivity_curve"]) >= 8


def test_api_parcels():
    """Smoke test GET /api/parcels."""
    res = client.get("/api/parcels")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 239


def test_phase_4d_hyperparameter_differentiation():
    """Phase 4d Item 1: Fitting with oracle vs farm-only vs dense hyperparameters
    gives DIFFERENT predictive means and variances for the same sensors.
    """
    from backend.sim.parcels import ParcelNetwork
    from backend.sim.sensors import SensorNetwork
    from backend.model.gp import UniversalKrigingGP

    parcels_net = ParcelNetwork()
    sensors_net = SensorNetwork()
    st_coords = np.array([[s.x, s.y, s.rel_elev] for s in sensors_net.stations])
    parcel_coords = np.array([[p.centroid_x, p.centroid_y, p.rel_elev] for p in parcels_net.parcels])
    dummy_temps = np.array([1.8, 0.4, 3.1, 2.1, 2.3, 3.1])

    gp_farm = UniversalKrigingGP(l_xy=527.2, l_h=148.9, sigma_f=0.585, sigma_n=0.101, sigma_scale=1.0)
    gp_dense = UniversalKrigingGP(l_xy=2500.0, l_h=15.4, sigma_f=0.930, sigma_n=0.050, sigma_scale=1.0)
    gp_oracle = UniversalKrigingGP(l_xy=750.0, l_h=40.0, sigma_f=0.850, sigma_n=0.300, sigma_scale=1.0)

    gp_farm.fit(st_coords, dummy_temps)
    mu_farm, sig_farm = gp_farm.predict(parcel_coords, recalibrate=False)

    gp_dense.fit(st_coords, dummy_temps)
    mu_dense, sig_dense = gp_dense.predict(parcel_coords, recalibrate=False)

    gp_oracle.fit(st_coords, dummy_temps)
    mu_oracle, sig_oracle = gp_oracle.predict(parcel_coords, recalibrate=False)

    # Assert predictions genuinely differ across configurations
    assert np.max(np.abs(mu_farm - mu_dense)) > 0.05, "Farm and dense means must differ"
    assert np.max(np.abs(sig_farm - sig_oracle)) > 0.05, "Farm and oracle sigmas must differ"
    assert np.max(np.abs(sig_farm - sig_dense)) > 0.05, "Farm and dense sigmas must differ"


def test_phase_4d_counts_reconciliation_and_catalog_identities():
    """Phase 4d Item 3: Count reconciliation and catalog identities.
    For every strategy, sensor set, night type and hero night:
      - frozen = TP + FN
      - protected = TP + FP
      - loss = FN * LOSS_PER_PARCEL
      - protection_cost = protected * COST_PER_PARCEL
      - total_expenditure = loss + protection_cost
    Catalog:
      - training inversion frozen parcel-nights = 1,170 over 37 nights
      - training ordinary frozen parcel-nights = 673 over 3 nights
    """
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    assert eval_file.exists(), "eval.json must exist"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    # Catalog verification
    cat = ev.get("catalog_verification", {})
    assert cat["training_inversion_frozen_parcel_nights"] == 1170
    assert cat["training_ordinary_frozen_parcel_nights"] == 673
    assert cat["training_total_frozen_parcel_nights"] == 1843
    assert cat["hero_frozen_parcels"] == 49
    assert cat["grand_total_frozen_parcel_nights"] == 1892

    roi = ev["disaggregated_roi_and_sensitivities"]
    cost_p = roi["parameters"]["cost_per_parcel_night"]
    loss_p = roi["parameters"]["loss_per_parcel"]

    # Test training counts identities
    t8 = roi["training_counts_k8"]["all_59_nights"]
    for strat_key, counts in t8.items():
        assert counts["frozen"] == counts["TP"] + counts["FN"], f"{strat_key}: frozen != TP + FN"
        assert counts["protected"] == counts["TP"] + counts["FP"], f"{strat_key}: protected != TP + FP"
        assert counts["crop_loss"] == pytest.approx(counts["FN"] * loss_p, abs=1e-2), f"{strat_key}: loss != FN * L"
        assert counts["protection_cost"] == pytest.approx(counts["protected"] * cost_p, abs=1e-2), f"{strat_key}: cost != prot * C"
        assert counts["total_expenditure"] == pytest.approx(counts["crop_loss"] + counts["protection_cost"], abs=1e-2)

    # Test hero night counts identities
    hero = roi["hero_night_showcase"]
    for strat_key in ["strategy_a", "strategy_b1", "strategy_b2", "strategy_c_k6", "strategy_c_k8"]:
        h = hero[strat_key]
        assert h["frozen"] == h["TP"] + h["FN"], f"Hero {strat_key}: frozen != TP + FN"
        assert h["protected"] == h["TP"] + h["FP"], f"Hero {strat_key}: protected != TP + FP"
        assert h["crop_loss"] == pytest.approx(h["FN"] * loss_p, abs=1e-2)
        assert h["protection_cost"] == pytest.approx(h["protected"] * cost_p, abs=1e-2)
        assert h["total_expenditure"] == pytest.approx(h["crop_loss"] + h["protection_cost"], abs=1e-2)


def test_phase_4d_sensitivity_protected_set_varies():
    """Phase 4d Item 5: In sensitivity analysis, as p* = C/L changes,
    the protected set changes, so K=8 total expenditure is NOT linear in L.
    """
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    curve = ev["disaggregated_roi_and_sensitivities"]["loss_to_cost_sensitivity_curve"]
    prot_counts = [pt["mean_n_protected_c"] for pt in curve]
    # As ratio increases (p* decreases), protected count must strictly increase
    assert len(set(prot_counts)) > 1, "Protected count must vary as p* varies"
    for i in range(len(prot_counts) - 1):
        assert prot_counts[i] <= prot_counts[i + 1], "Protected count should increase as p* decreases"


def test_phase_4d_strategies_b3_and_b4():
    """Phase 4d Item 6: Margin rule (B3) and local-knowledge DEM rule (B4)
    are evaluated and conform to accounting identities.
    """
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    hero = ev["disaggregated_roi_and_sensitivities"]["hero_night_showcase"]
    assert "strategy_b3_m3_k6" in hero
    assert "strategy_b4_m3_k6" in hero

    b4_m3 = hero["strategy_b4_m3_k6"]
    # B4 protects 24 DEM pocket parcels: catches 21 pocket frozen, misses 28 hollow frozen
    assert b4_m3["TP"] == 21
    assert b4_m3["FN"] == 28
    assert b4_m3["protected"] == 24


def test_phase_4e_reweighted_and_sensitivity_exact_matches():
    """Phase 4e Items 2 & 3:
    - Synthetic catalog row equals the catalog-weighted average of type-wise costs.
    - B1 at the default ratio equals the aggregate (€5,515,950 / 59 = €93,491).
    - Every displayed figure is integer-count derived.
    """
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    roi = ev["disaggregated_roi_and_sensitivities"]
    scenarios = roi["climatological_frequency_reweighting"]
    synth_sc = scenarios[0]
    assert synth_sc["scenario"] == "Synthetic Catalog Distribution"

    inputs = synth_sc["inputs_mean_exp_by_type"]
    cat_weights = {"ordinary": 3.0 / 59.0, "inversion": 37.0 / 59.0, "no_frost": 19.0 / 59.0}
    for s in ["B1", "B2", "B3_m3", "B4_m3", "B4p_m3", "C_k6", "C_k8"]:
        expected_synth = (
            cat_weights["ordinary"] * inputs["ordinary"][s]
            + cat_weights["inversion"] * inputs["inversion"][s]
            + cat_weights["no_frost"] * inputs["no_frost"][s]
        )
        assert synth_sc["weighted_expenditure"][s] == pytest.approx(expected_synth, abs=0.05)
        # Check it matches the 59-night aggregate per-night mean
        all_59_mean = roi["training_counts_k8"]["all_59_nights"][s]["mean_expenditure_per_night"]
        assert synth_sc["weighted_expenditure"][s] == pytest.approx(all_59_mean, abs=0.05)

    # Check B1 sensitivity match at default ratio (€4500 / €350 = 12.86)
    curve = roi["loss_to_cost_sensitivity_curve"]
    default_pt = next(pt for pt in curve if abs(pt["crop_loss_eur"] - 4500) < 1e-2)
    assert default_pt["mean_exp_b1"] == pytest.approx(93490.68, abs=0.05)
    assert default_pt["mean_exp_b1"] == pytest.approx(5515950.0 / 59.0, abs=0.05)

    # Check integer derivations
    for pt in curve:
        L = pt["crop_loss_eur"]
        C = 350.0
        # B1: 3 ordinary nights protect 239 parcels each (717 parcel-nights); 37 inversion nights lose 1170 parcels
        b1_tot = 717 * C + 1170 * L
        assert pt["mean_exp_b1"] == pytest.approx(b1_tot / 59.0, abs=0.05)


def test_phase_4f_exact_integer_accounting_and_linearity():
    """Phase 4f:
    - B1 is strictly linear in L: (717 * C + 1170 * L) / 59
    - For every row in sensitivity curve: protected * C <= total cost, and total equals protection + loss from counts
    - Recall equals TP / (TP + FN) across sequential table, calibration comparison, and ROI
    - Reweighted scenarios equal weighted sum of per-type integer rows
    """
    import json
    from backend.app.config import DATA_DIR

    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    # 1. B1 strict linearity & cost identity checks
    curve = ev["disaggregated_roi_and_sensitivities"]["loss_to_cost_sensitivity_curve"]
    C = 350.0
    for row in curve:
        L = row["crop_loss_eur"]
        expected_b1 = (717 * C + 1170 * L) / 59.0
        assert row["mean_exp_b1"] == pytest.approx(expected_b1, abs=0.01)

        # C(K=8) integer derivations
        n_prot = row["integer_protected_c_total"]
        n_fn = row["integer_damaged_c_total"]
        total_c = n_prot * C + n_fn * L
        assert n_prot * C <= total_c
        assert row["mean_exp_c_k8"] == pytest.approx(total_c / 59.0, abs=0.01)

    # 2. Recall identity TP / (TP + FN) across calibration table
    calib = ev["calibrations_side_by_side"]
    for m in ["farm_b_default", "dense_a_sensitivity", "oracle_sensitivity"]:
        c_m = calib[m]["integer_counts"]
        rec = calib[m]["recall"]
        expected_pooled = c_m["TP"] / (c_m["TP"] + c_m["FN"])
        assert rec["pooled_p_star"] == pytest.approx(expected_pooled, abs=1e-4)
        assert c_m["TP"] + c_m["FN"] == 1843
        fin = calib[m]["financial_breakdown_eur"]
        assert fin["protection_cost"] == c_m["protected"] * 350
        assert fin["crop_loss"] == c_m["FN"] * 4500
        assert fin["total_expenditure"] == fin["protection_cost"] + fin["crop_loss"]

    # 3. Recall identity TP / (TP + FN) across sequential table
    seq = ev["sequential_k6_to_k12_trajectory"]
    for row in seq:
        hero_all = row["hero_night"]["all_frozen"]
        tp = hero_all["TP"]
        fn = hero_all["FN"]
        if tp + fn > 0:
            assert hero_all["rec_p_star"] == pytest.approx(tp / (tp + fn), abs=1e-3)

    # 4. Seasonal scenario weighted sum identity
    scenarios = ev["disaggregated_roi_and_sensitivities"]["climatological_frequency_reweighting"]
    for sc in scenarios:
        w = sc["weights"]
        inputs = sc["inputs_per_type_per_night_eur"]
        exp = sc["weighted_expenditure_eur"]
        for s in ["B1", "B2", "B3_m3", "B4_m3", "C_k6", "C_k8"]:
            expected = (
                w["ordinary"] * inputs["ordinary"][s]
                + w["inversion"] * inputs["inversion"][s]
                + w["no_frost"] * inputs["no_frost"][s]
            )
            # weighted_expenditure is rounded to 2 dp; tolerance 0.01 covers rounding
            assert exp[s] == pytest.approx(expected, abs=0.01)



