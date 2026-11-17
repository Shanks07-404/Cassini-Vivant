"""API smoke tests for /api/config and /api/parcels."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Cassini Vivant"
    assert data["honesty_badge"] == "SIMULATED DATA"
    assert data["is_simulated_data"] is True
    assert "center" in data
    assert data["center"]["lat"] == pytest.approx(47.81, abs=0.01)


def test_api_parcels():
    response = client.get("/api/parcels")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["simulated"] is True
    assert data["metadata"]["disclaimer"] == "SIMULATED DATA"
    features = data["features"]
    assert 220 <= len(features) <= 260

    # Test parcel property contracts
    first_feat = features[0]
    props = first_feat["properties"]
    assert props["id"].startswith("P")
    assert "elevation" in props
    assert "rel_elev" in props
    assert "slope_deg" in props
    assert "is_frost_pocket" in props
    assert "value" in props
    assert 0.0 <= props["value"] <= 1.0


def test_api_parcels_value_override():
    response = client.get("/api/parcels?placeholder_val=0.42")
    assert response.status_code == 200
    data = response.json()
    assert data["features"][0]["properties"]["value"] == 0.42
