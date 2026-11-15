"""Script to precompute microclimate replay frames to disk for fast playback."""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import DATA_DIR
from backend.sim.microclimate import MicroclimateSimulator
from backend.sim.nights import build_night_profiles, list_night_ids
from backend.sim.parcels import ParcelNetwork
from backend.sim.sensors import SensorNetwork


def precompute_replays(num_nights: int = 60) -> None:
    output_dir = DATA_DIR / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing simulation environment...")
    parcels_net = ParcelNetwork()
    sensors_net = SensorNetwork()
    sim = MicroclimateSimulator(parcels=parcels_net, sensors=sensors_net)

    profiles = build_night_profiles()
    night_ids = list_night_ids()[:num_nights]

    print(f"Precomputing replay data for {len(night_ids)} nights into {output_dir}...")
    start_t = time.time()

    for nid in night_ids:
        prof = profiles[nid]
        p_temps, s_readings, r_temps, frozen_set = sim.simulate_night(prof)

        # Build hourly frame records
        frames = []
        for h_idx, hour in enumerate(sim.HOURS):
            hourly_frozen = [
                sim.parcels[p_idx].id
                for p_idx in range(len(sim.parcels))
                if p_temps[h_idx, p_idx] < 0.0
            ]
            frames.append({
                "hour_idx": h_idx,
                "hour": hour,
                "hour_label": f"{hour:02d}:00",
                "regional_temp": round(float(r_temps[h_idx]), 2),
                "frozen_count": len(hourly_frozen),
                "frozen_parcels": hourly_frozen,
                "sensor_readings": {
                    sensors_net.stations[s_idx].id: round(float(s_readings[h_idx, s_idx]), 2)
                    for s_idx in range(len(sensors_net.stations))
                },
                "parcel_temperatures": {
                    sim.parcels[p_idx].id: round(float(p_temps[h_idx, p_idx]), 2)
                    for p_idx in range(len(sim.parcels))
                },
            })

        # Count pockets vs non-pockets in overall frozen set
        pocket_set = {p.id for p in sim.parcels if p.is_valley_bottom}
        frozen_pockets = [pid for pid in frozen_set if pid in pocket_set]
        frozen_outside = [pid for pid in frozen_set if pid not in pocket_set]

        payload = {
            "night_id": nid,
            "is_hero_night": prof.is_hero_night,
            "t_min_regional": prof.t_min_regional,
            "t_max_regional": prof.t_max_regional,
            "hours": sim.HOURS,
            "total_parcels": len(sim.parcels),
            "all_night_frozen_count": len(frozen_set),
            "all_night_frozen_pockets_count": len(frozen_pockets),
            "all_night_frozen_outside_pockets_count": len(frozen_outside),
            "all_night_frozen_parcels": frozen_set,
            "frames": frames,
        }

        target_file = output_dir / f"{nid}.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    elapsed = time.time() - start_t
    print(f"Precomputed {len(night_ids)} nights in {elapsed:.2f}s ({elapsed / len(night_ids) * 1000:.1f}ms/night).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nights", type=int, default=60, help="Number of nights to precompute")
    args = parser.parse_args()
    precompute_replays(num_nights=args.nights)
