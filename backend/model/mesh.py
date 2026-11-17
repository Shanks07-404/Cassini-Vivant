"""Delaunay triangulation mesh ('Cassini Triangles') connecting active weather stations."""

from typing import Any, Dict, List
import numpy as np
from scipy.spatial import Delaunay

from backend.sim.sensors import SensorNetwork


def compute_delaunay_mesh(sensors_net: SensorNetwork) -> Dict[str, Any]:
    """Compute Delaunay triangulation of active stations and return GeoJSON lines."""
    stations = sensors_net.stations
    if len(stations) < 3:
        return {
            "type": "FeatureCollection",
            "metadata": {"triangle_count": 0, "edge_count": 0},
            "features": [],
        }

    points_metric = np.array([[s.x, s.y] for s in stations])
    tri = Delaunay(points_metric)

    # Extract unique undirected edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            p1, p2 = simplex[i], simplex[(i + 1) % 3]
            edge = tuple(sorted((p1, p2)))
            edges.add(edge)

    features: List[Dict[str, Any]] = []
    for idx, (i1, i2) in enumerate(edges):
        s1, s2 = stations[i1], stations[i2]
        dist_m = float(np.hypot(s1.x - s2.x, s1.y - s2.y))

        features.append({
            "type": "Feature",
            "id": f"mesh_edge_{idx:03d}",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [s1.lon, s1.lat],
                    [s2.lon, s2.lat],
                ],
            },
            "properties": {
                "station_a": s1.id,
                "station_b": s2.id,
                "length_m": round(dist_m, 1),
                "is_cassini_triangle_edge": True,
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "station_count": len(stations),
            "triangle_count": len(tri.simplices),
            "edge_count": len(edges),
            "simulated": True,
        },
        "features": features,
    }
