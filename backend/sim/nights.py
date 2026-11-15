"""Catalog of simulated meteorological nights for Cassini Vivant.

Generates 60 reproducible nights with varied severity:
- night_001: Designated HERO radiation frost night.
- nights 002-020: Mild spring/autumn nights (zero freeze).
- nights 021-040: Moderate katabatic inversion nights (valley freeze).
- nights 041-055: Strong inversion nights (valley + tributary drainage freeze).
- nights 056-060: Severe cold snaps (regional temp near or below freezing).
"""

from typing import Dict, List, Optional
from backend.sim.microclimate import MicroclimateProfile


def build_night_profiles() -> Dict[str, MicroclimateProfile]:
    """Generate deterministic catalog of 60 nights."""
    profiles: Dict[str, MicroclimateProfile] = {}

    # Hero Night 001
    profiles["night_001"] = MicroclimateProfile(
        night_id="night_001",
        is_hero_night=True,
        t_min_regional=2.4,        # Regional ambient stays > 2°C (NO regional frost alert!)
        t_max_regional=7.2,        # Evening starting temp at 20:00
        alpha_pool=4.2,            # Pronounced cold-air pooling in valley floor
        h0_pool=16.0,              # Inversion layer depth (16m)
        alpha_drain=2.5,           # Strong drainage along tributary hollows
        clear_calm_factor=1.0,     # Perfect clear sky & calm radiative conditions
        spatial_seed=1001,
    )

    # Nights 002 - 020: Mild nights (zero frost)
    for i in range(2, 21):
        nid = f"night_{i:03d}"
        seed = 2000 + i
        t_min = 5.5 + (i % 6) * 0.9
        profiles[nid] = MicroclimateProfile(
            night_id=nid,
            is_hero_night=False,
            t_min_regional=t_min,
            t_max_regional=t_min + 5.5,
            alpha_pool=2.0 + (i % 3) * 0.4,
            h0_pool=14.0,
            alpha_drain=1.0,
            clear_calm_factor=0.6,
            spatial_seed=seed,
        )

    # Nights 021 - 040: Moderate inversion nights (valley floor frost)
    for i in range(21, 41):
        nid = f"night_{i:03d}"
        seed = 3000 + i
        t_min = 2.8 + (i % 4) * 0.3
        profiles[nid] = MicroclimateProfile(
            night_id=nid,
            is_hero_night=False,
            t_min_regional=t_min,
            t_max_regional=t_min + 5.0,
            alpha_pool=3.6 + (i % 3) * 0.3,
            h0_pool=15.0,
            alpha_drain=1.5,
            clear_calm_factor=0.9,
            spatial_seed=seed,
        )

    # Nights 041 - 055: Strong inversion nights
    for i in range(41, 56):
        nid = f"night_{i:03d}"
        seed = 4000 + i
        t_min = 2.0 + (i % 3) * 0.3
        profiles[nid] = MicroclimateProfile(
            night_id=nid,
            is_hero_night=False,
            t_min_regional=t_min,
            t_max_regional=t_min + 4.8,
            alpha_pool=4.4 + (i % 2) * 0.3,
            h0_pool=16.5,
            alpha_drain=2.2,
            clear_calm_factor=0.95,
            spatial_seed=seed,
        )

    # Nights 056 - 060: Severe cold snap
    for i in range(56, 61):
        nid = f"night_{i:03d}"
        seed = 5000 + i
        t_min = 0.5 - (i - 56) * 0.4
        profiles[nid] = MicroclimateProfile(
            night_id=nid,
            is_hero_night=False,
            t_min_regional=t_min,
            t_max_regional=t_min + 4.0,
            alpha_pool=4.5,
            h0_pool=18.0,
            alpha_drain=2.6,
            clear_calm_factor=1.0,
            spatial_seed=seed,
        )

    return profiles


_PROFILES_CACHE = build_night_profiles()


def get_night_profile(night_id: str) -> Optional[MicroclimateProfile]:
    return _PROFILES_CACHE.get(night_id)


def list_night_ids() -> List[str]:
    return list(_PROFILES_CACHE.keys())


def get_hero_night_profile() -> MicroclimateProfile:
    return _PROFILES_CACHE["night_001"]
