"""Tests for Phase 2: Microclimate physics, sensor network, nights catalog, and replay."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.sim.microclimate import MicroclimateSimulator
from backend.sim.nights import build_night_profiles, get_hero_night_profile
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork

client = TestClient(app)


def test_sensors_default_placement():
    """Verify 6 default stations are strictly outside frost pockets and lowest is just above threshold."""
    sensors_net = SensorNetwork()
    stations = sensors_net.stations

    assert len(stations) == 6
    for s in stations:
        assert s.id.startswith("S")
        assert s.is_pocket is False
        assert s.rel_elev > 8.0, f"Station {s.id} is inside frost pocket: rel_elev={s.rel_elev}"

    # Lowest station relative elevation must be just above the 8m threshold (10m - 16m)
    rel_elevs = [s.rel_elev for s in stations]
    lowest_rel = min(rel_elevs)
    assert 10.0 <= lowest_rel <= 16.0, f"Lowest station relative elevation {lowest_rel} not in [10, 16]m"

    # Higher stations should span mid-slopes and ridges
    highest_rel = max(rel_elevs)
    assert highest_rel > 100.0, f"Highest station relative elevation {highest_rel} too low"


def test_hero_night_regional_and_sensors_stay_above_freezing():
    """Requirement B: Regional temperature and ALL sensor readings stay > 0 C all night on night_001."""
    sim = MicroclimateSimulator()
    prof = get_hero_night_profile()
    p_temps, s_readings, r_temps, frozen_set = sim.simulate_night(prof)

    # 1. Regional temp stays > 0 C
    assert float(np_min := r_temps.min()) > 0.0, f"Regional temperature dropped below freezing: {np_min}"
    assert np_min >= 2.0, f"Regional minimum expected >= 2.0 C, got {np_min}"

    # 2. All 6 sensor readings stay > 0 C all night
    assert float(s_min := s_readings.min()) > 0.0, f"Sensor reading dropped below freezing: {s_min}"


def test_hero_night_frozen_set_composition():
    """Requirement B: Frozen set includes valley pocket AND at least ~10 parcels outside flagged pocket."""
    sim = MicroclimateSimulator()
    prof = get_hero_night_profile()
    p_temps, s_readings, r_temps, frozen_set = sim.simulate_night(prof)

    pocket_ids = {p.id for p in sim.parcels if p.is_valley_bottom}
    frozen_in_pocket = [pid for pid in frozen_set if pid in pocket_ids]
    frozen_outside_pocket = [pid for pid in frozen_set if pid not in pocket_ids]

    # Total frozen set substantial
    assert len(frozen_set) >= 30, f"Expected >= 30 frozen parcels, got {len(frozen_set)}"
    # Covers majority of valley pocket
    assert len(frozen_in_pocket) >= 15, f"Expected >= 15 pocket parcels frozen, got {len(frozen_in_pocket)}"
    # At least ~10 parcels outside flagged pocket (drained by tributary hollows)
    assert len(frozen_outside_pocket) >= 10, (
        f"Expected >= 10 outside-pocket frozen parcels, got {len(frozen_outside_pocket)}"
    )

    # Demo parcel P214 must freeze
    assert "P214" in frozen_set, "Demo parcel P214 was expected to freeze on hero night_001"


def test_nights_determinism():
    """Verify that simulating the same night profile twice produces bit-for-bit identical results."""
    sim = MicroclimateSimulator()
    prof = get_hero_night_profile()

    p_temps1, s_read1, r_temps1, frozen1 = sim.simulate_night(prof)
    p_temps2, s_read2, r_temps2, frozen2 = sim.simulate_night(prof)

    assert (p_temps1 == p_temps2).all()
    assert (s_read1 == s_read2).all()
    assert (r_temps1 == r_temps2).all()
    assert frozen1 == frozen2


def test_api_sensors():
    """Test GET /api/sensors, POST /api/sensors, DELETE /api/sensors, and reset."""
    # List
    res = client.get("/api/sensors")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 6
    assert len(data["stations"]) == 6

    # Add
    add_res = client.post("/api/sensors", json={"lat": 47.815, "lon": 3.805, "name": "Test Probe"})
    assert add_res.status_code == 200
    added_id = add_res.json()["station"]["id"]

    # Verify count increased
    list_res2 = client.get("/api/sensors")
    assert list_res2.json()["count"] == 7

    # Delete
    del_res = client.delete(f"/api/sensors/{added_id}")
    assert del_res.status_code == 200

    # Reset
    reset_res = client.post("/api/sensors/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["count"] == 6


def test_api_replay():
    """Test GET /api/replay/night_001 and GET /api/nights."""
    # List nights
    nights_res = client.get("/api/nights")
    assert nights_res.status_code == 200
    nights_data = nights_res.json()
    assert nights_data["count"] == 60
    assert nights_data["hero_night"] == "night_001"

    # Get hero replay
    replay_res = client.get("/api/replay/night_001")
    assert replay_res.status_code == 200
    replay_data = replay_res.json()
    assert replay_data["night_id"] == "night_001"
    assert replay_data["is_hero_night"] is True
    assert len(replay_data["frames"]) == 13
    assert replay_data["all_night_frozen_count"] >= 35
    assert "P214" in replay_data["all_night_frozen_parcels"]

    # Verify hourly frame structure
    frame0 = replay_data["frames"][0]
    assert frame0["hour"] == 20
    assert "parcel_temperatures" in frame0
    assert "sensor_readings" in frame0
    assert "frozen_parcels" in frame0
