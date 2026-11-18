"""Unit tests for synthetic terrain and parcel generation in Phase 0 & Phase 1."""

import pytest
import numpy as np

from backend.sim.terrain import SyntheticTerrain
from backend.sim.parcels import ParcelNetwork


def test_terrain_elevation_range():
    terrain = SyntheticTerrain()
    # Check bounds across grid
    xs = np.linspace(-1800, 1800, 20)
    ys = np.linspace(-1800, 1800, 20)
    gx, gy = np.meshgrid(xs, ys)
    z = terrain.elevation(gx, gy)

    assert np.all(z >= 125.0), f"Minimum elevation {z.min()} too low"
    assert np.all(z <= 340.0), f"Maximum elevation {z.max()} too high"


def test_terrain_valley_relative_elevation():
    terrain = SyntheticTerrain()
    # At the stream, relative elevation should be near 0
    for y in [-1000.0, 0.0, 1000.0]:
        sx = terrain.stream_x(y)
        rel_z = terrain.relative_elevation(sx, y)
        assert rel_z == pytest.approx(0.0, abs=1e-3)

    # Further up the slopes, relative elevation should be substantially positive
    for y in [-1000.0, 0.0, 1000.0]:
        rel_east = terrain.relative_elevation(1200.0, y)
        rel_west = terrain.relative_elevation(-1200.0, y)
        assert rel_east > 40.0
        assert rel_west > 40.0


def test_terrain_coordinate_roundtrip():
    terrain = SyntheticTerrain()
    test_points = [(0.0, 0.0), (1200.0, -800.0), (-1500.0, 1400.0)]
    for x, y in test_points:
        lon, lat = terrain.to_wgs84(x, y)
        x_rec, y_rec = terrain.to_local(lon, lat)
        assert x_rec == pytest.approx(x, abs=0.1)
        assert y_rec == pytest.approx(y, abs=0.1)
        # Lat/lon near Chablis (47.81 N, 3.80 E)
        assert 47.75 < lat < 47.87
        assert 3.74 < lon < 3.86


def test_terrain_slope_and_aspect():
    terrain = SyntheticTerrain()
    slope, sin_a, cos_a = terrain.slope_and_aspect(500.0, 500.0)
    assert 0.0 <= slope <= 45.0
    assert -1.0 <= sin_a <= 1.0
    assert -1.0 <= cos_a <= 1.0
    assert (sin_a ** 2 + cos_a ** 2) == pytest.approx(1.0, abs=1e-3)


def test_parcel_network_generation():
    network = ParcelNetwork(seed=42)
    # Check count is ~250
    assert 220 <= len(network) <= 260, f"Expected ~250 parcels, got {len(network)}"

    # Check properties of all parcels
    valley_count = 0
    for p in network.parcels:
        assert p.id.startswith("P")
        assert len(p.id) == 4  # P### format
        assert len(p.polygon_wgs84) >= 3
        assert p.area_ha > 0.1
        assert p.elevation > 125.0
        assert p.rel_elev >= 0.0
        assert -1.0 <= p.aspect_sin <= 1.0
        assert -1.0 <= p.aspect_cos <= 1.0
        if p.is_valley_bottom:
            valley_count += 1

    # Ensure we have low-lying valley parcels for frost pooling
    assert valley_count >= 10, f"Expected at least 10 valley floor parcels, found {valley_count}"


def test_parcel_determinism():
    """Verify that same seed produces exactly identical parcel IDs, coordinates, and frost flags."""
    net1 = ParcelNetwork(seed=42)
    net2 = ParcelNetwork(seed=42)

    assert len(net1) == len(net2)
    for p1, p2 in zip(net1.parcels, net2.parcels):
        assert p1.id == p2.id
        assert p1.name == p2.name
        assert p1.centroid_lon == p2.centroid_lon
        assert p1.centroid_lat == p2.centroid_lat
        assert p1.elevation == p2.elevation
        assert p1.rel_elev == p2.rel_elev
        assert p1.is_valley_bottom == p2.is_valley_bottom
        assert p1.polygon_wgs84 == p2.polygon_wgs84


def test_parcel_geojson_structure():
    network = ParcelNetwork(seed=42)
    geojson = network.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert geojson["metadata"]["simulated"] is True
    assert geojson["metadata"]["disclaimer"] == "SIMULATED DATA"
    assert len(geojson["features"]) == len(network)

    feat0 = geojson["features"][0]
    assert feat0["type"] == "Feature"
    assert "geometry" in feat0
    assert feat0["geometry"]["type"] == "Polygon"
    assert "properties" in feat0
    assert feat0["properties"]["id"].startswith("P")
    assert "rel_elev" in feat0["properties"]
    assert "slope_deg" in feat0["properties"]
    assert "value" in feat0["properties"]
    assert "is_frost_pocket" in feat0["properties"]
    assert 0.0 <= feat0["properties"]["value"] <= 1.0
