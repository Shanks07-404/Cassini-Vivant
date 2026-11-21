"""Tests for Phase 5 Gemini Assistant, 6 Spec Tools, and Dossier generation."""

import json
import os
import pytest
from backend.app.api.assistant_api import ChatRequest, chat, execute_tool, get_dossier_html
from backend.app.config import DATA_DIR
from backend.assistant.retrieval import retriever


def test_retriever_fictionalized_rules():
    """Verify retriever only indexes fictionalized demo rules and glossary."""
    retriever._load_and_index()
    doc_ids = {c.doc_id for c in retriever.chunks}
    
    # Must NOT contain real-looking or old rule names
    assert "AOP-CHAB-2024" not in doc_ids
    assert "INAO-ENV-01" not in doc_ids
    assert "CAS-SOP-FROST" not in doc_ids
    
    # Must contain fictionalized rulebooks
    assert "DEMO-APPELLATION-RULES" in doc_ids or "DEMO-AOP-TERROIR" in doc_ids
    assert "DEMO-FROST-PROTECTION" in doc_ids
    assert "GLOSSARY" in doc_ids
    assert "adaptation_mechanisms" in doc_ids


def test_get_parcel_risk_model_output_and_night_differentiation():
    """Verify get_parcel_risk returns GP model outputs for a NON-sensor parcel (P214) and differentiates nights.
    
    Must return mu, calibrated sigma, p_frost, protect_decision (p_frost >= p*), and sensor_set_used.
    Must NOT use is_frost_pocket for risk. P214 is in the combe cold pool and is NOT a station parcel.
    """
    # 1. Hero night (severe inversion: P214 experiences severe cold pooling)
    r_hero = execute_tool("get_parcel_risk", {"parcel_id": "P214", "night_id": "night_001", "sensor_set": "K8"})
    assert r_hero["parcel_id"] == "P214"
    assert "mu_temperature_c" in r_hero
    assert "sigma_calibrated_c" in r_hero
    assert "p_frost" in r_hero
    assert "action_threshold_p_star" in r_hero
    assert "protect_decision" in r_hero
    assert "sensor_set_used" in r_hero
    assert "K8" in r_hero["sensor_set_used"]

    # On hero night, P214 freezes severely (-3.79 C true), so mu is negative and p_frost is high (>0.5)
    assert r_hero["mu_temperature_c"] < 0.0
    assert r_hero["p_frost"] > 0.50
    assert r_hero["protect_decision"] is True

    # 2. No-frost night (night_002 has regional temp ~6.5 C)
    r_nofrost = execute_tool("get_parcel_risk", {"parcel_id": "P214", "night_id": "night_002", "sensor_set": "K8"})
    assert r_nofrost["parcel_id"] == "P214"
    assert r_nofrost["mu_temperature_c"] > 4.0
    assert r_nofrost["p_frost"] < 0.01
    assert r_nofrost["protect_decision"] is False

    # Night differentiation assertion
    assert abs(r_hero["p_frost"] - r_nofrost["p_frost"]) > 0.50
    assert r_hero["protect_decision"] != r_nofrost["protect_decision"]


def test_six_spec_tools_contracts():
    """Verify the canonical set of 6 spec tools execute and conform to required contracts."""
    # 1. get_parcel_risk
    r1 = execute_tool("get_parcel_risk", {"parcel_id": "P050", "night_id": "night_001"})
    assert r1["parcel_id"] == "P050"
    assert "p_frost" in r1

    # 2. get_sensor_status
    r2 = execute_tool("get_sensor_status", {"sensor_set": "K8", "night_id": "night_001", "hour": 4})
    assert r2["sensor_set"] == "K8"
    assert r2["station_count"] == 8
    assert len(r2["stations"]) == 8
    assert "temp_c" in r2["stations"][0]

    # 3. search_appellation_rules
    r3 = execute_tool("search_appellation_rules", {"query": "candles and smoke"})
    assert "query" in r3
    assert len(r3["matched_sections"]) > 0
    top_doc = r3["matched_sections"][0]["doc_id"]
    assert any(pref in top_doc for pref in ["DEMO-", "adaptation", "GLOSSARY"])

    # 4. get_night_summary (must report TP, FN, FP, TN, and costs)
    r4 = execute_tool("get_night_summary", {"night_id": "night_001", "strategy": "C_k8"})
    assert r4["night_id"] == "night_001"
    assert "confusion_matrix" in r4
    assert r4["confusion_matrix"]["TP"] == 35
    assert r4["confusion_matrix"]["FN"] == 14
    assert r4["TP"] == 35
    assert r4["FN"] == 14
    assert "protection_cost_eur" in r4
    assert "crop_loss_eur" in r4
    assert "total_expenditure_eur" in r4
    assert r4["total_expenditure_eur"] == r4["protection_cost_eur"] + r4["crop_loss_eur"]

    # 5. draft_dossier
    r5 = execute_tool("draft_dossier", {"lang": "en"})
    assert "watermark" in r5
    assert "economic_framework" in r5
    assert "regulatory_and_environmental" in r5

    # 6. calculate_roi_strategy (reads sensitivity table, never scales linearly in L)
    r6_low = execute_tool("calculate_roi_strategy", {"strategy": "C_k8", "loss_per_parcel": 1750.0})
    r6_high = execute_tool("calculate_roi_strategy", {"strategy": "C_k8", "loss_per_parcel": 10500.0})
    
    assert r6_low["total_protected_parcel_nights"] < r6_high["total_protected_parcel_nights"]
    loss_ratio = 10500.0 / 1750.0  # 6.0x
    cost_ratio = r6_high["total_expenditure_59_nights_eur"] / r6_low["total_expenditure_59_nights_eur"]
    assert cost_ratio < loss_ratio  # Non-linear because protected set expands

    # 7. get_sensor_sequence
    r7 = execute_tool("get_sensor_sequence", {})
    assert "trajectory" in r7
    assert len(r7["trajectory"]) == 7
    assert r7["trajectory"][0]["k"] == 6


def test_assistant_offline_fallback_no_api_key(monkeypatch):
    """Verify chat endpoint falls back to deterministic cached answers when no API key is provided."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from backend.app.config import settings
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")

    req = ChatRequest(message="What happened on the hero night night_001?")
    res = chat(req)

    assert res["is_offline"] is True
    assert "banner" in res
    assert "night_001" in res["answer"] or "hero" in res["answer"].lower()
    assert len(res["citations"]) > 0
    for c in res["citations"]:
        assert not c["doc_id"].startswith("AOP-CHAB")
        assert not c["doc_id"].startswith("INAO")


def test_dossier_html_watermark_and_limitations():
    """Verify dossier HTML contains DEMO DATA watermark, limitations, and no regulatory compliance claims."""
    resp = get_dossier_html(lang="en")
    html = resp.body.decode("utf-8")

    assert "DEMO DATA" in html
    assert "SIMULATION ONLY" in html
    assert "Limitations" in html
    assert "simulated" in html.lower()
    assert "AOP-CHAB-2024" not in html
    assert "INAO-ENV-01" not in html

    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)
    p_star = ev["disaggregated_roi_and_sensitivities"]["parameters"]["optimal_action_threshold_p_star"]
    p_star_str = f"{p_star * 100:.2f}%"
    assert p_star_str in html


def test_ui_dossier_and_cached_answers_derive_strictly_from_eval_json():
    """Verify all cached assistant answers and dossier figures are derived strictly from eval.json."""
    eval_file = DATA_DIR / "eval.json"
    cached_file = DATA_DIR / "cached_answers.json"
    assert eval_file.exists()
    assert cached_file.exists()

    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)
    with open(cached_file, "r", encoding="utf-8") as f:
        cached_data = json.load(f)

    cached_by_id = {ans["id"]: ans for ans in cached_data["answers"]}

    # 1. Optimal p* in cached answers
    p_star = ev["disaggregated_roi_and_sensitivities"]["parameters"]["optimal_action_threshold_p_star"]
    p_star_pct = f"{p_star * 100:.2f}%"
    hero_ans = cached_by_id["hero_night_diagnostic"]["answer"]
    assert p_star_pct in hero_ans

    # 2. Hero night frozen count (49)
    hero_n_frozen = ev["catalog_verification"]["hero_frozen_parcels"]
    assert str(hero_n_frozen) in hero_ans

    # 3. Strategy B1 cost in cached answers
    roi_ans = cached_by_id["strategy_comparison"]["answer"]
    mean_b1 = ev["disaggregated_roi_and_sensitivities"]["training_counts_k8"]["all_59_nights"]["B1"]["mean_expenditure_per_night"]
    assert f"{mean_b1:,.2f}" in roi_ans or f"{round(mean_b1):,}" in roi_ans


def test_model_selection_never_touches_night_001():
    """Verify that model fitting, hyperparameter calibration, and next-best sensor selection
    never touch the hero night night_001. All model choices are fit strictly on training nights 002-060.
    """
    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    # 1. Catalog verification: hero night is strictly 1 night, training nights are 59
    cat = ev["catalog_verification"]
    assert cat["hero_frozen_parcels"] == 49
    assert cat["training_total_frozen_parcel_nights"] == 1843
    nt = ev["night_types_catalog"]
    assert nt["total_nights"] == 60
    assert nt["ordinary_frost"]["count"] == 3
    assert nt["inversion_frost"]["count"] == 38  # 1 hero + 37 training
    assert nt["no_frost"]["count"] == 19
    # 59 training nights (3 + 37 + 19)
    training_nights_count = nt["ordinary_frost"]["count"] + (nt["inversion_frost"]["count"] - 1) + nt["no_frost"]["count"]
    assert training_nights_count == 59

    # 2. Denominators: inversion nights is 37, ordinary is 3, total training is 59 (never 60)
    denom = ev["denominators_and_reconciliation"]
    assert denom["n_nights_inversion_frost"] == 37
    assert denom["n_nights_ordinary_frost"] == 3
    assert denom["n_frost_training_nights"] == 40

    # 3. Decision-focused sequential placement was evaluated on training nights 002-060
    seq = ev["sequential_k6_to_k12_trajectory"]
    for row in seq:
        tr = row["training_nights"]
        # Total parcel nights evaluated is exactly 59 * 239 = 14,101 (never includes night_001)
        int_c = tr["integer_counts"]
        total_eval = int_c["TP"] + int_c["FN"] + int_c["FP"] + int_c["TN"]
        assert total_eval == 14101, f"Expected 14,101 parcel-nights (59*239), got {total_eval}"
        # Hero night is evaluated separately in a dedicated block
        hero = row["hero_night"]["all_frozen"]
        assert hero["TP"] + hero["FN"] == 49


def test_eval_json_frozen_hash_integrity():
    """Verify that eval.json is permanently frozen and its SHA-256 hash matches the authoritative commit."""
    import hashlib
    eval_file = DATA_DIR / "eval.json"
    assert eval_file.exists(), "eval.json must exist"
    with open(eval_file, "rb") as f:
        file_bytes = f.read()
    current_hash = hashlib.sha256(file_bytes).hexdigest().upper()
    expected_hash = "0CE04B40723F4A0FC161C2BD8A810AD2CD02B78A81DF5FCE5C163D6C71358AC6"
    assert current_hash == expected_hash, (
        f"eval.json has been modified! Expected hash {expected_hash}, got {current_hash}."
    )

