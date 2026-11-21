"""Tests for Phase 5 Assistant, Rulebook Retrieval, and HTML Dossier endpoints."""

import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import DATA_DIR
from backend.assistant.retrieval import retriever

client = TestClient(app)


def test_rulebook_retriever_indexing():
    """Verify retriever indexes all fictionalized demo rule documents and finds citations."""
    assert len(retriever.chunks) >= 3

    # Query for water aspersion
    chunks = retriever.search("water sprinkling aspersion", top_k=2)
    assert len(chunks) > 0
    doc_ids = [c["doc_id"] for c in chunks]
    assert any("DEMO" in d or "adaptation" in d for d in doc_ids)
    assert "citation" in chunks[0]

    # Query for cost-loss economic threshold
    chunks_sop = retriever.search("optimal cost loss decision rule p*", top_k=2)
    assert len(chunks_sop) > 0
    assert any("DEMO-FROST-PROTECTION" in c["doc_id"] for c in chunks_sop)


def test_assistant_chat_offline_fallback_and_banner():
    """Verify POST /api/assistant/chat returns disclosure banner and figures matching eval.json."""
    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    hero_n = ev["catalog_verification"]["hero_frozen_parcels"]

    res = client.post("/api/assistant/chat", json={"message": "What happened on the hero night night_001?", "lang": "en"})
    assert res.status_code == 200
    data = res.json()

    assert "banner" in data
    assert "AI-generated decision support advice" in data["banner"]
    assert "answer" in data
    assert str(hero_n) in data["answer"]  # 49 parcels
    assert "citations" in data
    assert len(data["citations"]) > 0
    assert any("DEMO" in c["doc_id"] for c in data["citations"])


def test_assistant_chat_roi_figures_derived():
    """Verify chat ROI response contains numbers matching eval.json."""
    res = client.post("/api/assistant/chat", json={"message": "What is the ROI of adding K=8 sensors?", "lang": "en"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "P050" in data["answer"] or "P202" in data["answer"]
    assert "€" in data["answer"] or "%" in data["answer"]


def test_dossier_html_watermark_and_figures():
    """Verify GET /api/dossier/html includes watermark and figures matching eval.json."""
    eval_file = DATA_DIR / "eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        ev = json.load(f)

    res = client.get("/api/dossier/html")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text

    # Watermark assertion
    assert "DEMO DATA — SIMULATION ONLY" in html

    # Disclaimer and Limitations
    assert "Limitations" in html
    assert "DEMO-AOP-TERROIR" in html or "DEMO" in html

    # Figures from eval.json
    t8 = ev["disaggregated_roi_and_sensitivities"]["training_counts_k8"]["all_59_nights"]
    b1_mean = f"{t8['B1']['mean_expenditure_per_night']:,.2f}"
    assert b1_mean in html or "93,490.68" in html
