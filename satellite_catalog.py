import math
import time

import numpy as np


RE_KM = 6371.0
MU = 398600.4418
OMEGA = 7.2921159e-5


CONSTELLATIONS_CFG = {
    "Starlink": {
        "color": [56, 189, 248],
        "shells": [
            {"count": 1500, "planes": 72, "alt": 550, "inc": 53.0, "phase": 0.37},
            {"count": 720, "planes": 36, "alt": 570, "inc": 70.0, "phase": 0.51},
            {"count": 360, "planes": 18, "alt": 560, "inc": 97.6, "phase": 0.23},
            {"count": 500, "planes": 40, "alt": 540, "inc": 43.0, "phase": 0.41},
        ],
    },
    "Amazon LEO": {
        "color": [251, 146, 60],
        "shells": [{"count": 900, "planes": 36, "alt": 630, "inc": 51.9, "phase": 0.51}],
    },
    "OneWeb": {
        "color": [248, 113, 113],
        "shells": [{"count": 648, "planes": 18, "alt": 1200, "inc": 87.9, "phase": 0.22}],
    },
    "Iridium": {
        "color": [74, 222, 128],
        "shells": [{"count": 66, "planes": 6, "alt": 780, "inc": 86.4, "phase": 0.17}],
    },
    "GPS/Galileo": {
        "color": [250, 204, 21],
        "shells": [{"count": 96, "planes": 12, "alt": 20200, "inc": 56.0, "phase": 0.31}],
    },
}


def make_catalog():
    sats = []
    sid = 1
    for cname, cfg in CONSTELLATIONS_CFG.items():
        for shell_idx, shell in enumerate(cfg["shells"]):
            count = int(shell["count"])
            planes = int(shell["planes"])
            sats_per_plane = math.ceil(count / planes)
            alt = float(shell["alt"])
            semi_major = RE_KM + alt
            mean_motion = math.sqrt(MU / semi_major**3)
            inc = math.radians(shell["inc"])
            for k in range(count):
                plane = k // sats_per_plane
                slot = k % sats_per_plane
                sats.append({
                    "id": sid,
                    "name": f"{cname}-{sid}",
                    "constellation": cname,
                    "color": cfg["color"],
                    "shell": shell_idx + 1,
                    "raan": 2 * math.pi * plane / planes,
                    "phase0": (
                        2 * math.pi * slot / sats_per_plane
                        + float(shell["phase"]) * plane
                        + shell_idx * 0.7
                    ),
                    "inc": inc,
                    "inc_deg": float(shell["inc"]),
                    "alt_km": alt,
                    "a": semi_major,
                    "n": mean_motion,
                })
                sid += 1
    return sats


def propagate_satellite(sat, sim_sec):
    u = sat["phase0"] + sat["n"] * sim_sec
    cu, su = math.cos(u), math.sin(u)
    co, so = math.cos(sat["raan"]), math.sin(sat["raan"])
    ci, si = math.cos(sat["inc"]), math.sin(sat["inc"])

    xo = sat["a"] * cu
    yo = sat["a"] * su

    xeci = co * xo - so * ci * yo
    yeci = so * xo + co * ci * yo
    zeci = si * yo

    theta = OMEGA * sim_sec
    ct, st = math.cos(theta), math.sin(theta)
    x = ct * xeci + st * yeci
    y = -st * xeci + ct * yeci
    z = zeci

    r = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * RE_KM * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def latlon_to_xyz_km(lat_deg, lon_deg, radius_km):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.array(
        [
            radius_km * np.cos(lat) * np.cos(lon),
            radius_km * np.cos(lat) * np.sin(lon),
            radius_km * np.sin(lat),
        ],
        dtype=float,
    )


def current_sim_seconds(speed=10.0):
    return time.time() * float(speed)


def compute_corridor_box(pairs_df):
    """Rectangle corridor: left edge = easternmost DP, right edge = westernmost DT."""
    dp_lons = list(pairs_df["dp_lon"])
    dp_lats = list(pairs_df["dp_lat"])
    dt_lons = list(pairs_df["dt_lon"])
    dt_lats = list(pairs_df["dt_lat"])

    lon_left  = max(dp_lons)
    lon_right = min(dt_lons)
    if lon_left > lon_right:
        lon_left, lon_right = lon_right, lon_left

    return {
        "lon_left":  lon_left,
        "lon_right": lon_right,
        "lat_bot":   min(dp_lats + dt_lats),
        "lat_top":   max(dp_lats + dt_lats),
    }


def select_satellites_for_pairs(
    pairs_df,
    catalog,
    constellation="Starlink",
    margin_deg=5.0,
    sim_sec=None,
):
    if sim_sec is None:
        sim_sec = current_sim_seconds()

    corridor = compute_corridor_box(pairs_df)
    dp_lat = float(np.mean(pairs_df["dp_lat"]))
    dp_lon = float(np.mean(pairs_df["dp_lon"]))
    dt_lat = float(np.mean(pairs_df["dt_lat"]))
    dt_lon = float(np.mean(pairs_df["dt_lon"]))
    center_lat = (dp_lat + dt_lat) / 2.0
    center_lon = (dp_lon + dt_lon) / 2.0

    lat_pad = margin_deg

    # Single pass: propagate all constellation sats, split into all / corridor
    all_constellation = []
    corridor_pool = []
    for sat in catalog:
        if sat["constellation"] != constellation:
            continue
        lat, lon = propagate_satellite(sat, sim_sec)
        row = {**sat, "lat": lat, "lon": lon}
        all_constellation.append(row)
        if (
            corridor["lon_left"] <= lon <= corridor["lon_right"]
            and (corridor["lat_bot"] - lat_pad) <= lat <= (corridor["lat_top"] + lat_pad)
        ):
            corridor_pool.append(row)

    # Fallback if corridor is too sparse
    if len(corridor_pool) < 5:
        corridor_pool = sorted(
            all_constellation,
            key=lambda s: haversine_km(center_lat, center_lon, s["lat"], s["lon"]),
        )[:80]

    # L1: 2 satellites closest to DP centroid
    l1_pool = sorted(corridor_pool, key=lambda s: haversine_km(dp_lat, dp_lon, s["lat"], s["lon"]))
    layer1 = l1_pool[:2]
    used_ids = {s["id"] for s in layer1}

    # L2: 3 satellites closest to DT centroid, no overlap with L1
    l2_candidates = [s for s in corridor_pool if s["id"] not in used_ids]
    if len(l2_candidates) < 3:
        l2_candidates = sorted(
            [s for s in all_constellation if s["id"] not in used_ids],
            key=lambda s: haversine_km(dt_lat, dt_lon, s["lat"], s["lon"]),
        )
    layer2 = sorted(l2_candidates, key=lambda s: haversine_km(dt_lat, dt_lon, s["lat"], s["lon"]))[:3]

    selected = []
    for local_idx, sat in enumerate(layer1):
        selected.append({**sat, "local_idx": local_idx, "layer": "L1"})
    for offset, sat in enumerate(layer2):
        selected.append({**sat, "local_idx": 2 + offset, "layer": "L2"})

    return {
        "selected":         selected,
        "candidates":       corridor_pool,
        "all_constellation": all_constellation,
        "corridor":         corridor,
        "sim_sec":          sim_sec,
    }
