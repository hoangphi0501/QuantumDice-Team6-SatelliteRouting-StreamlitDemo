import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from routing_backend import build_and_solve_gibbs_cpu
from satellite_catalog import (
    CONSTELLATIONS_CFG, make_catalog, propagate_satellite, select_satellites_for_pairs,
)


st.set_page_config(
    page_title="Quantum Dice (Team 6) — Satellite Routing",
    page_icon="🛰️",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="metric-container"] {
    background: #0f1c2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 14px !important;
  }
  .stDataFrame { border-radius: 6px; overflow: hidden; }
  div[data-testid="stExpander"] > details {
    border: 1px solid #1e3a5f;
    border-radius: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
_C_DP        = [37,  99,  235]
_C_DT        = [22,  163,  74]
_C_L1        = [245, 158,  11]
_C_L2        = [168,  85, 247]
_C_CORRIDOR  = [100, 180, 255]
_C_IN_BOX    = [120, 200, 255]
_C_OUT_BOX   = [100, 100, 140]


# ── Catalog (cached) ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_catalog():
    return make_catalog()


@st.cache_data(show_spinner=False)
def propagate_constellation_cached(constellation: str, time_bucket: int) -> list:
    """Propagate all sats of one constellation. Cache key changes every 30 s."""
    sim_sec = float(time_bucket) * 30.0 * 10.0   # 10× orbital speed
    result = []
    for sat in get_catalog():
        if sat["constellation"] != constellation:
            continue
        lat, lon = propagate_satellite(sat, sim_sec)
        result.append({**sat, "lat": lat, "lon": lon})
    return result


# ── Data helpers ───────────────────────────────────────────────────────────────
def default_pairs(n_pairs):
    base = [
        {"pair": 1, "dp_lon": -52.7126, "dp_lat": 47.5615, "dt_lon": -3.1883, "dt_lat": 55.9533},
        {"pair": 2, "dp_lon": -63.5752, "dp_lat": 44.6488, "dt_lon": -0.1278, "dt_lat": 51.5074},
    ]
    rows = []
    for i in range(n_pairs):
        row = dict(base[i]) if i < len(base) else {
            "pair": i + 1, "dp_lon": -55.0 - i, "dp_lat": 45.0 + 0.5 * i,
            "dt_lon": -5.0 + i, "dt_lat": 52.0 + 0.5 * i,
        }
        row["pair"] = i + 1
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_pairs(df, n_pairs):
    df = df.copy().head(n_pairs)
    if len(df) < n_pairs:
        df = pd.concat([df, default_pairs(n_pairs).iloc[len(df):]], ignore_index=True)
    df["pair"] = np.arange(1, n_pairs + 1)
    for col in ["dp_lon", "dp_lat", "dt_lon", "dt_lat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["dp_lon", "dp_lat", "dt_lon", "dt_lat"]].isna().any().any():
        raise ValueError("All DP/DT longitude and latitude values must be numeric.")
    if not ((df["dp_lat"].between(-90, 90)).all() and (df["dt_lat"].between(-90, 90)).all()):
        raise ValueError("Latitude must be between -90 and 90.")
    if not ((df["dp_lon"].between(-180, 180)).all() and (df["dt_lon"].between(-180, 180)).all()):
        raise ValueError("Longitude must be between -180 and 180.")
    return df


# ── DataFrame builders ─────────────────────────────────────────────────────────
def ground_points_df(pairs_df):
    rows = []
    for row in pairs_df.itertuples(index=False):
        rows += [
            {"name": f"DP{int(row.pair)} — Departure", "label": f"DP{int(row.pair)}",
             "kind": "DP", "lon": float(row.dp_lon), "lat": float(row.dp_lat),
             "color": _C_DP, "glow_color": _C_DP + [50]},
            {"name": f"DT{int(row.pair)} — Destination", "label": f"DT{int(row.pair)}",
             "kind": "DT", "lon": float(row.dt_lon), "lat": float(row.dt_lat),
             "color": _C_DT, "glow_color": _C_DT + [50]},
        ]
    return pd.DataFrame(rows)


def satellite_points_df(selected):
    rows = []
    for sat in selected:
        is_l1   = sat["layer"] == "L1"
        color   = _C_L1 if is_l1 else _C_L2
        local   = sat["local_idx"] if is_l1 else sat["local_idx"] - 2
        layer_s = "L1" if is_l1 else "L2"
        rows.append({
            "name":      f"{layer_s}-{local} · SAT #{sat['id']} · {sat['alt_km']:.0f}km alt",
            "label":     f"{layer_s}-{local}",
            "layer":     layer_s,
            "lon":       float(sat["lon"]),
            "lat":       float(sat["lat"]),
            "alt_km":    float(sat["alt_km"]),
            "color":     color,
            "glow_color": color + [45],
        })
    return pd.DataFrame(rows)


def corridor_polygon_df(corridor):
    ll, lr = corridor["lon_left"],  corridor["lon_right"]
    lb, lt = corridor["lat_bot"],   corridor["lat_top"]
    return pd.DataFrame([{
        "polygon": [[ll, lb], [lr, lb], [lr, lt], [ll, lt], [ll, lb]],
        "name":    "Selection Corridor",
    }])


def corridor_dashed_border_df(corridor, dash: float = 2.0, gap: float = 1.2):
    """Simulate a dashed rectangle border as short PathLayer segments."""
    ll, lr = corridor["lon_left"],  corridor["lon_right"]
    lb, lt = corridor["lat_bot"],   corridor["lat_top"]
    step = dash + gap

    def _edge_dashes(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return []
        segs, t = [], 0.0
        while t < length:
            t0 = t / length
            t1 = min(t + dash, length) / length
            segs.append({
                "path": [
                    [p1[0] + t0 * dx, p1[1] + t0 * dy],
                    [p1[0] + t1 * dx, p1[1] + t1 * dy],
                ],
                "color": [100, 180, 255, 190],
            })
            t += step
        return segs

    rows = (
        _edge_dashes([ll, lb], [lr, lb])   # bottom
        + _edge_dashes([lr, lb], [lr, lt]) # right
        + _edge_dashes([lr, lt], [ll, lt]) # top
        + _edge_dashes([ll, lt], [ll, lb]) # left
    )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def arc_segments_df(pairs_df, selected, routes):
    # Count how many pairs share each L1→L2 ISL link
    isl_usage: dict = {}
    for route in routes:
        key = (int(route["layer1"]), int(route["layer2"]))
        isl_usage[key] = isl_usage.get(key, 0) + 1

    rows = []
    for route in routes:
        m    = int(route["m"])
        pair = pairs_df.iloc[m]
        l1i  = int(route["layer1"])
        l2i  = int(route["layer2"])
        s1   = selected[l1i]
        s2   = selected[l2i]

        n_share   = isl_usage.get((l1i, l2i), 1)
        congested = n_share > 1
        isl_src   = [255, 70,  70,  230] if congested else (_C_L1 + [220])
        isl_tgt   = [255, 140, 30,  230] if congested else (_C_L2 + [220])
        isl_name  = f"Pair {m+1}: L1 → L2 ⚠ CONGESTED ×{n_share}" if congested else f"Pair {m+1}: L1 → L2"

        rows += [
            {"name": f"Pair {m+1}: DP → L1",
             "src_lon": float(pair["dp_lon"]), "src_lat": float(pair["dp_lat"]),
             "tgt_lon": float(s1["lon"]),      "tgt_lat": float(s1["lat"]),
             "src_color": _C_DP + [220], "tgt_color": _C_L1 + [220], "width": 4},
            {"name": isl_name,
             "src_lon": float(s1["lon"]), "src_lat": float(s1["lat"]),
             "tgt_lon": float(s2["lon"]), "tgt_lat": float(s2["lat"]),
             "src_color": isl_src, "tgt_color": isl_tgt,
             "width": 7 if congested else 4},
            {"name": f"Pair {m+1}: L2 → DT",
             "src_lon": float(s2["lon"]),      "src_lat": float(s2["lat"]),
             "tgt_lon": float(pair["dt_lon"]), "tgt_lat": float(pair["dt_lat"]),
             "src_color": _C_L2 + [220], "tgt_color": _C_DT + [220], "width": 4},
        ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def labels_df(pairs_df, selected):
    rows = []
    for row in pairs_df.itertuples(index=False):
        rows += [
            {"label": f"DP{int(row.pair)}", "lon": float(row.dp_lon), "lat": float(row.dp_lat),
             "color": [180, 210, 255, 230]},
            {"label": f"DT{int(row.pair)}", "lon": float(row.dt_lon), "lat": float(row.dt_lat),
             "color": [160, 255, 195, 230]},
        ]
    for sat in selected:
        is_l1  = sat["layer"] == "L1"
        local  = sat["local_idx"] if is_l1 else sat["local_idx"] - 2
        rows.append({
            "label": f"{'L1' if is_l1 else 'L2'}-{local}",
            "lon":   float(sat["lon"]),
            "lat":   float(sat["lat"]),
            "color": [255, 225, 140, 230] if is_l1 else [215, 160, 255, 230],
        })
    return pd.DataFrame(rows)


def all_constellation_df(all_sats, selected_ids):
    """All background constellation satellites — uniform style, selected ones excluded."""
    if not all_sats:
        return pd.DataFrame()
    rows = []
    for sat in all_sats:
        if sat["id"] in selected_ids:
            continue
        rows.append({
            "name": f"{sat['name']} · {sat['alt_km']:.0f}km",
            "lon":  float(sat["lon"]),
            "lat":  float(sat["lat"]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Result renderers ───────────────────────────────────────────────────────────
def render_latency_breakdown(breakdown):
    if not breakdown:
        st.info("No latency breakdown available.")
        return
    try:
        first = next(iter(breakdown.values()))
        if isinstance(first, dict):
            # Nested dict: {segment: {component: value_s, ...}}
            rows = []
            for seg_key, sub in breakdown.items():
                row = {"Segment": str(seg_key)}
                for k, v in sub.items():
                    try:
                        row[k] = float(v) * 1000
                    except (TypeError, ValueError):
                        row[k] = str(v)
                rows.append(row)
            df = pd.DataFrame(rows)
            num_cols = [c for c in df.columns if c != "Segment"
                        and pd.api.types.is_numeric_dtype(df[c])]
            col_cfg = {c: st.column_config.NumberColumn(c, format="%.4f ms") for c in num_cols}
            st.dataframe(df, column_config=col_cfg, use_container_width=True, hide_index=True)
            return
    except (StopIteration, AttributeError):
        pass
    # Flat dict: {component: value_s}
    rows = []
    for k, v in breakdown.items():
        try:
            rows.append({"Component": k, "ms": float(v) * 1000})
        except (TypeError, ValueError):
            continue
    if not rows:
        st.json(breakdown)
        return
    df = pd.DataFrame(rows)
    max_ms = float(df["ms"].max()) if not df.empty else 1.0
    st.dataframe(
        df,
        column_config={
            "ms": st.column_config.ProgressColumn(
                "Latency (ms)",
                format="%.4f",
                min_value=0,
                max_value=max_ms * 1.1 if max_ms > 0 else 1.0,
            )
        },
        use_container_width=True,
        hide_index=True,
    )


def render_selected_vars(repaired_solution):
    rows = []
    for var, value in repaired_solution.items():
        if int(value) != 1:
            continue
        try:
            vtype, m, (s1, s2) = var
            is_xp = vtype == "xp"
            rows.append({
                "Pair":  m + 1,
                "Link":  "🔵→🟡  DP → L1 SAT" if is_xp else "🟡→🟣  L1 → L2 SAT",
                "From":  s1,
                "To":    s2,
            })
        except Exception:
            pass
    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values(["Pair", "Link"]),
            use_container_width=True,
            hide_index=True,
        )


# ── Map rendering ──────────────────────────────────────────────────────────────
def render_map(
    pairs_df,
    selected=None,
    routes=None,
    corridor=None,
    all_sats=None,
    show_all=True,
    height=500,
    trail_frames=None,
):
    selected = selected or []
    routes   = routes   or []
    selected_ids = {s["id"] for s in selected}

    ground_df = ground_points_df(pairs_df)
    sat_df    = satellite_points_df(selected) if selected else pd.DataFrame()
    arc_df    = arc_segments_df(pairs_df, selected, routes) if (selected and routes) else pd.DataFrame()
    lbl_df    = labels_df(pairs_df, selected)

    all_lons = list(ground_df["lon"]) + (list(sat_df["lon"]) if not sat_df.empty else [])
    all_lats = list(ground_df["lat"]) + (list(sat_df["lat"]) if not sat_df.empty else [])
    view = pdk.ViewState(
        latitude=float(np.mean(all_lats)),
        longitude=float(np.mean(all_lons)),
        zoom=2.5, pitch=0,
    )

    layers = []

    # 0 ── All constellation background — comet tail (ghost frames) + current frame
    if show_all:
        # Ghost trail: older frames rendered first (behind), progressively more transparent
        _TRAIL_ALPHAS = [18, 38, 65]  # oldest → newest ghost
        if trail_frames:
            n_trail = len(trail_frames)
            for ti, frame_positions in enumerate(trail_frames):
                alpha = _TRAIL_ALPHAS[max(0, ti - (3 - n_trail))]
                if frame_positions:
                    trail_df = pd.DataFrame(frame_positions)
                    if not trail_df.empty:
                        layers.append(pdk.Layer(
                            "ScatterplotLayer", data=trail_df,
                            get_position="[lon, lat]",
                            get_fill_color=[150, 185, 255, alpha],
                            get_radius=1,
                            radius_min_pixels=1,
                            radius_max_pixels=4,
                            pickable=False,
                        ))
        # Current frame: brightest layer
        if all_sats:
            bg_df = all_constellation_df(all_sats, selected_ids)
            if not bg_df.empty:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=bg_df,
                    get_position="[lon, lat]",
                    get_fill_color=[150, 185, 255, 120],
                    get_radius=1,
                    radius_min_pixels=2,
                    radius_max_pixels=6,
                    pickable=True,
                ))

    # 1 ── Corridor: fill mờ + viền nét đứt
    if corridor:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=corridor_polygon_df(corridor),
            get_polygon="polygon",
            filled=True, stroked=False,
            get_fill_color=_C_CORRIDOR + [18],
            pickable=True,
        ))
        dash_df = corridor_dashed_border_df(corridor)
        if not dash_df.empty:
            layers.append(pdk.Layer(
                "PathLayer",
                data=dash_df,
                get_path="path",
                get_color="color",
                get_width=2,
                width_min_pixels=2,
                pickable=False,
            ))

    # 2 ── Ground glow + solid
    layers += [
        pdk.Layer("ScatterplotLayer", data=ground_df,
                  get_position="[lon, lat]", get_fill_color="glow_color",
                  get_radius=180000, pickable=False),
        pdk.Layer("ScatterplotLayer", data=ground_df,
                  get_position="[lon, lat]", get_fill_color="color",
                  get_radius=90000, pickable=True),
    ]

    # 3 ── Selected satellite glow + solid (L1 amber, L2 violet)
    if not sat_df.empty:
        for lyr, c in [("L1", _C_L1), ("L2", _C_L2)]:
            sub = sat_df[sat_df["layer"] == lyr]
            if sub.empty:
                continue
            layers += [
                pdk.Layer("ScatterplotLayer", data=sub,
                          get_position="[lon, lat]", get_fill_color=c + [40],
                          get_radius=150000, pickable=False),
                pdk.Layer("ScatterplotLayer", data=sub,
                          get_position="[lon, lat]", get_fill_color=c + [255],
                          get_radius=65000, pickable=True),
            ]

    # 4 ── Route arcs (great-circle, gradient; congested ISL = red + wider)
    if not arc_df.empty:
        layers.append(pdk.Layer(
            "ArcLayer", data=arc_df,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[tgt_lon, tgt_lat]",
            get_source_color="src_color",
            get_target_color="tgt_color",
            get_width="width",
            width_min_pixels=3,
            great_circle=True, pickable=True,
        ))

    # 5 ── Floating labels
    if not lbl_df.empty:
        layers.append(pdk.Layer(
            "TextLayer", data=lbl_df,
            get_position="[lon, lat]", get_text="label",
            get_size=14, get_color="color",
            get_pixel_offset=[0, -22],
            billboard=True, pickable=False,
        ))

    st.pydeck_chart(
        pdk.Deck(
            map_style="dark",
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "{name}"},
        ),
        use_container_width=True,
        height=height,
    )

    # Legend
    st.markdown("""
    <div style="display:flex;gap:18px;padding:6px 2px 2px;flex-wrap:wrap;
                font-size:12.5px;color:#888;line-height:1.8;">
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
        background:#2563eb;margin-right:5px;vertical-align:middle;"></span>DP</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
        background:#16a34a;margin-right:5px;vertical-align:middle;"></span>DT</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
        background:#f59e0b;margin-right:5px;vertical-align:middle;"></span>L1 Satellite</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
        background:#a855f7;margin-right:5px;vertical-align:middle;"></span>L2 Satellite</span>
      <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
        background:rgba(150,185,255,0.55);margin-right:5px;vertical-align:middle;"></span>Other sats</span>
      <span><span style="display:inline-block;width:13px;height:10px;border-radius:2px;
        background:rgba(100,180,255,0.12);border:1.5px solid rgba(100,180,255,0.75);
        margin-right:5px;vertical-align:middle;"></span>Selection Corridor</span>
      <span><span style="display:inline-block;width:20px;height:3px;
        background:linear-gradient(to right,#ff4646,#ff8c1e);
        margin-right:5px;vertical-align:middle;border-radius:2px;"></span>Congested L1→L2</span>
    </div>
    """, unsafe_allow_html=True)


# ── Section header helper ──────────────────────────────────────────────────────
def _sec(n, title):
    st.markdown(
        f'<p style="font-size:18px;font-weight:700;margin:22px 0 6px 0;">'
        f'<span style="background:#1e3a5f;color:#60a5fa;font-size:13px;'
        f'font-weight:700;padding:2px 8px;border-radius:12px;margin-right:8px;">{n}</span>'
        f'{title}</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<h1 style="margin-bottom:2px;">🛰️ Quantum Dice (Team 6) — Satellite Routing</h1>'
    '<p style="color:#888;margin-top:0;">CPU Gibbs p-bit QUBO optimiser · corridor satellite selection · real-time tracking</p>',
    unsafe_allow_html=True,
)

catalog = get_catalog()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scenario")
    n_pairs       = st.number_input("Number of DP/DT pairs", 1, 20, 2, 1)
    constellation = st.selectbox("Satellite constellation", list(CONSTELLATIONS_CFG.keys()))
    margin_deg    = st.slider("Latitude corridor margin (°)", 0.0, 30.0, 5.0, 1.0,
                              help="Padding added above/below the corridor lat bounds.")

    st.markdown("### 📐 QUBO Weights")
    wt   = st.number_input("Wt — latency weight",    0.0, value=1.0, step=0.1)
    wc   = st.number_input("Wc — congestion weight", 0.0, value=1.0, step=0.1)
    lam1 = st.number_input("λ₁ — begin penalty",     0.0, value=20.0, step=5.0)
    lam2 = st.number_input("λ₂ — equality penalty",  0.0, value=20.0, step=5.0)

    st.markdown("### 🔬 CPU Gibbs")
    seed       = st.number_input("Random seed",  0,   value=42,   step=1)
    num_chains = st.number_input("Chains",       10,  1000, 80,   10)
    num_steps  = st.number_input("Steps",        50,  5000, 800,  50)
    batch_size = st.number_input("Batch size",   1,   128,  8,    1)
    t_start    = st.number_input("T_start",      0.01, value=5.0, step=0.5)
    t_end      = st.number_input("T_end",        0.001, value=0.01, step=0.01, format="%.3f")

    st.markdown("### 🗺️ Visualisation")
    show_all_sats = st.checkbox("Show all constellation satellites", value=True)

# ── DP/DT state ────────────────────────────────────────────────────────────────
if "pairs_df" not in st.session_state or len(st.session_state["pairs_df"]) != int(n_pairs):
    st.session_state["pairs_df"] = default_pairs(int(n_pairs))

# ── Section 1 ──────────────────────────────────────────────────────────────────
_sec("1", "DP / DT Coordinates")
st.caption("Enter longitude / latitude for each pair. Routing is fixed: DP1 → DT1, DP2 → DT2 …")

pairs_input = st.data_editor(
    st.session_state["pairs_df"],
    use_container_width=True, hide_index=True, disabled=["pair"],
    column_config={
        "pair":   st.column_config.NumberColumn("Pair"),
        "dp_lon": st.column_config.NumberColumn("DP lon", format="%.5f"),
        "dp_lat": st.column_config.NumberColumn("DP lat", format="%.5f"),
        "dt_lon": st.column_config.NumberColumn("DT lon", format="%.5f"),
        "dt_lat": st.column_config.NumberColumn("DT lat", format="%.5f"),
    },
)

try:
    pairs_df = normalize_pairs(pairs_input, int(n_pairs))
    st.session_state["pairs_df"] = pairs_df
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ── Section 2 — persistent constellation & route map ──────────────────────────
_sec("2", "Constellation & Route Map")

# Reserve map slot HERE (visually above buttons) — filled after button handling
_map_slot = st.empty()

# ── Action buttons ─────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    select_btn = st.button("🛰️ Select Corridor Satellites", use_container_width=True)
with col_b:
    run_btn = st.button("⚡ Run CPU Gibbs Optimisation", type="primary", use_container_width=True)

if select_btn or run_btn:
    with st.spinner("Propagating catalog · selecting corridor satellites…"):
        sat_payload = select_satellites_for_pairs(
            pairs_df, catalog,
            constellation=constellation,
            margin_deg=float(margin_deg),
        )
    st.session_state["sat_payload"] = sat_payload

if run_btn:
    if "sat_payload" not in st.session_state:
        st.error("Please select satellites first.")
        st.stop()
    with st.spinner("Solving QUBO · repairing feasibility…"):
        result = build_and_solve_gibbs_cpu(
            pairs_df=pairs_df,
            selected_sats=st.session_state["sat_payload"]["selected"],
            seed=int(seed), num_chains=int(num_chains), num_steps=int(num_steps),
            batch_size=int(batch_size), t_start=float(t_start), t_end=float(t_end),
            wt=float(wt), wc=float(wc), lam1=float(lam1), lam2=float(lam2),
        )
    st.session_state["solve_result"] = result

# ── Retrieve updated state ─────────────────────────────────────────────────────
sat_payload = st.session_state.get("sat_payload", {})
selected    = sat_payload.get("selected", [])
corridor    = sat_payload.get("corridor")
result      = st.session_state.get("solve_result")

# Full constellation (cached, updates every 30 s)
time_bucket = int(time.time() / 30)
bg_all = propagate_constellation_cached(constellation, time_bucket) if show_all_sats else []

# Fill the map placeholder with the latest state
with _map_slot.container():
    render_map(
        pairs_df,
        selected=selected,
        routes=(result or {}).get("routes", []),
        corridor=corridor,
        all_sats=bg_all if show_all_sats else None,
        show_all=show_all_sats,
        height=550,
    )

# ── Section 3 — selection info cards (no duplicate map) ───────────────────────
if selected:
    _sec("3", "Selected Satellites & Corridor")
    info_a, info_b = st.columns([3, 2])
    with info_a:
        sat_display = satellite_points_df(selected)[["name", "layer", "lon", "lat", "alt_km"]].rename(
            columns={"name": "Satellite", "layer": "Layer", "lon": "Lon", "lat": "Lat", "alt_km": "Alt (km)"}
        )
        st.dataframe(sat_display, use_container_width=True, hide_index=True)
    with info_b:
        if corridor:
            st.markdown(
                f"""
                <div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:8px;padding:14px 16px;">
                  <p style="color:#60a5fa;font-weight:600;margin:0 0 10px 0;font-size:14px;">
                    📦 Corridor Bounds
                  </p>
                  <table style="width:100%;font-size:13px;color:#ccc;border-collapse:collapse;">
                    <tr><td style="padding:3px 0;color:#888;">Longitude</td>
                        <td style="text-align:right;color:#fff;">{corridor['lon_left']:.3f}° → {corridor['lon_right']:.3f}°</td></tr>
                    <tr><td style="padding:3px 0;color:#888;">Latitude</td>
                        <td style="text-align:right;color:#fff;">{corridor['lat_bot']:.3f}° → {corridor['lat_top']:.3f}°</td></tr>
                    <tr><td style="padding:3px 0;color:#888;">Pool size (corridor)</td>
                        <td style="text-align:right;color:#fff;">{len(sat_payload.get("candidates", []))} sats</td></tr>
                    <tr><td style="padding:3px 0;color:#888;">Constellation total</td>
                        <td style="text-align:right;color:#fff;">{len(bg_all)} sats</td></tr>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Section 4 ──────────────────────────────────────────────────────────────────
if result:
    _sec("4", "Gibbs Optimisation Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables",          result["n_vars"])
    c2.metric("Raw objective",      f"{result['raw_best_obj']:.4f}")
    c3.metric("Repaired objective", f"{result['repaired_obj']:.4f}")
    c4.metric("Runtime",            f"{result['runtime_sec']:.2f} s")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Device",             result["device"])
    c6.metric("Raw feasible",       f"{result['raw_feasibility']['p_feasible']:.0%}")
    c7.metric("Repaired feasible",  f"{result['repaired_feasibility']['p_feasible']:.0%}")
    c8.metric("Congestion",         f"{result['congestion']:.4f}")

    st.metric("Total latency", f"{result['total_latency']:.6f} s")

    lat_cost  = float(wt) * result["total_latency"]
    cong_cost = float(wc) * result["congestion"]

    st.markdown("---")
    tab_routes, tab_repair, tab_latency, tab_vars, tab_cost = st.tabs(
        ["🗺️ Routes", "🔧 Repair Report", "⏱️ Latency Breakdown", "🔢 Selected Variables", "📊 QUBO Cost"]
    )

    with tab_routes:
        cards_html = ""
        for route in result["routes"]:
            m   = int(route["m"])
            l1  = int(route["layer1"])
            l2  = int(route["layer2"]) - 2
            cards_html += f"""
            <div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:8px;
                        padding:12px 18px;margin:6px 0;display:flex;
                        align-items:center;gap:10px;flex-wrap:wrap;">
              <span style="background:#1e3a5f;color:#60a5fa;padding:2px 10px;
                           border-radius:10px;font-size:12px;font-weight:700;
                           min-width:52px;text-align:center;">Pair {m+1}</span>
              <span style="background:#1d4ed8;color:#fff;padding:5px 14px;
                           border-radius:6px;font-size:13px;font-weight:600;">DP{m+1}</span>
              <span style="color:#3b82f6;font-size:18px;font-weight:300;">→</span>
              <span style="background:#b45309;color:#fff;padding:5px 14px;
                           border-radius:6px;font-size:13px;font-weight:600;">L1-{l1}</span>
              <span style="color:#3b82f6;font-size:18px;font-weight:300;">→</span>
              <span style="background:#6d28d9;color:#fff;padding:5px 14px;
                           border-radius:6px;font-size:13px;font-weight:600;">L2-{l2}</span>
              <span style="color:#3b82f6;font-size:18px;font-weight:300;">→</span>
              <span style="background:#15803d;color:#fff;padding:5px 14px;
                           border-radius:6px;font-size:13px;font-weight:600;">DT{m+1}</span>
            </div>"""
        st.markdown(cards_html, unsafe_allow_html=True)

    with tab_repair:
        _ACTION_ICON = {
            "kept":             "🟢 kept",
            "added_missing":    "🟠 added",
            "reduced_multiple": "🔴 reduced",
        }
        repair_df = pd.DataFrame(result["repair_report"]).rename(columns={
            "m":                    "Pair",
            "chosen_s_layer1":      "L1 chosen",
            "chosen_sp_layer2":     "L2 chosen",
            "num_selected_s_before":"xp count",
            "num_selected_sp_before":"xs count",
            "xp_action":            "xp action",
            "xs_action":            "xs action",
        })
        for col in ["xp action", "xs action"]:
            if col in repair_df.columns:
                repair_df[col] = repair_df[col].map(
                    lambda v: _ACTION_ICON.get(v, v)
                )
        repair_df["Pair"] = repair_df["Pair"] + 1
        st.dataframe(repair_df, use_container_width=True, hide_index=True)

    with tab_latency:
        render_latency_breakdown(result["latency_breakdown"])

    with tab_vars:
        render_selected_vars(result["repaired_solution"])

    with tab_cost:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "⏱️ Latency cost",
            f"{lat_cost:.6f} s",
            delta=f"Wt {wt:.2f} × {result['total_latency']:.6f} s",
            delta_color="off",
        )
        cc2.metric(
            "🔄 Congestion cost",
            f"{cong_cost:.4f}",
            delta=f"Wc {wc:.2f} × {result['congestion']:.4f}",
            delta_color=("normal" if result["congestion"] > 0.5 else "inverse"),
        )
        cc3.metric(
            "📊 Total QUBO obj",
            f"{result['repaired_obj']:.6f}",
            delta=f"lat {lat_cost:.4f} + cong {cong_cost:.4f}",
            delta_color="off",
        )
        st.caption(
            f"QUBO = **Wt** × latency + **Wc** × congestion + λ₁ × begin + λ₂ × equality  "
            f"| λ₁ = {lam1}, λ₂ = {lam2}"
        )

# ── Section 5 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="background:linear-gradient(135deg,#0a1628,#0f2040);'
    'border:1px solid #1e4080;border-left:4px solid #3b82f6;'
    'border-radius:8px;padding:14px 18px;margin:16px 0 10px 0;">'
    '<p style="color:#60a5fa;font-weight:700;font-size:17px;margin:0 0 4px 0;">⚡ 5 · Live Satellite Tracking</p>'
    '<p style="color:#6b8fbd;font-size:13px;margin:0;">Satellites move along their orbits · routes are re-optimised automatically</p>'
    '</div>',
    unsafe_allow_html=True,
)

live_enabled = st.toggle("Enable Live Mode", value=st.session_state.get("live_mode", False))
st.session_state["live_mode"] = live_enabled

if not live_enabled:
    # Clear accumulated sim time and trail so next enable starts fresh
    for _k in ("live_sim_sec", "live_last_wall", "live_bg_trail"):
        st.session_state.pop(_k, None)
    st.caption("Toggle on to start live satellite animation and real-time route optimisation.")
else:
    lc1, lc2, lc3 = st.columns([2, 2, 1])
    with lc1:
        _SPEED_OPTIONS = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 500]
        orbit_speed = st.select_slider(
            "Orbit speed ×",
            options=_SPEED_OPTIONS,
            value=60,
            help="Simulation time multiplier. 1× = real time; 60× ≈ 1 Starlink orbit per 1.5 min.",
        )
        _real_s = 5730.0 / float(orbit_speed)  # Starlink 550km LEO period ≈ 95.5 min
        _om, _os = int(_real_s // 60), int(_real_s % 60)
        _orbit_str = f"{_om}m {_os}s" if _om > 0 else f"{_os}s"
        st.caption(f"🛰️ **{orbit_speed}×** · 1 Starlink orbit ≈ **{_orbit_str}** real time")
    with lc2:
        reopt_secs = st.number_input(
            "Re-optimise every (s)", min_value=5, max_value=120, value=10, step=5,
        )
    with lc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        force_reopt = st.button("▶ Now", use_container_width=True, help="Force immediate re-optimisation")

    # Timing info
    now       = time.time()
    last_reopt   = st.session_state.get("live_last_reopt", 0.0)
    elapsed_reopt = now - last_reopt
    remaining_reopt = max(0.0, float(reopt_secs) - elapsed_reopt)

    stat_a, stat_b = st.columns(2)
    stat_a.caption(f"Last re-optimised **{elapsed_reopt:.1f}s** ago")
    stat_b.caption(f"Next re-optimisation in **{remaining_reopt:.1f}s**")
    st.progress(min(1.0, elapsed_reopt / max(1, float(reopt_secs))))

    # Accumulate sim time incrementally — no jump when orbit_speed changes mid-run
    if "live_last_wall" not in st.session_state:
        st.session_state["live_sim_sec"] = 0.0
        st.session_state["live_last_wall"] = now
    else:
        dt = max(0.0, min(now - st.session_state["live_last_wall"], 5.0))
        st.session_state["live_sim_sec"] = st.session_state.get("live_sim_sec", 0.0) + dt * float(orbit_speed)
        st.session_state["live_last_wall"] = now
    live_sim_sec = st.session_state["live_sim_sec"]

    with st.spinner("Propagating satellite positions…"):
        live_payload = select_satellites_for_pairs(
            pairs_df, catalog,
            constellation=constellation,
            margin_deg=float(margin_deg),
            sim_sec=live_sim_sec,
        )

    # Re-optimise if interval passed or forced
    needs_reopt = force_reopt or (elapsed_reopt >= float(reopt_secs))
    if needs_reopt:
        with st.spinner("Re-optimising routes…"):
            live_result = build_and_solve_gibbs_cpu(
                pairs_df=pairs_df,
                selected_sats=live_payload["selected"],
                seed=int(seed), num_chains=int(num_chains), num_steps=int(num_steps),
                batch_size=int(batch_size), t_start=float(t_start), t_end=float(t_end),
                wt=float(wt), wc=float(wc), lam1=float(lam1), lam2=float(lam2),
            )
        st.session_state["live_result"]     = live_result
        st.session_state["live_last_reopt"] = time.time()

    live_result = st.session_state.get("live_result")

    # Freshly propagated sats at live_sim_sec — fall back to empty if module not reloaded yet
    live_bg = live_payload.get("all_constellation", [])

    # Update comet-tail trail (keep last 3 frames of background sat positions)
    _trail = st.session_state.get("live_bg_trail", [])
    if live_bg and show_all_sats:
        _snapshot = [{"lon": float(s["lon"]), "lat": float(s["lat"])} for s in live_bg]
        _trail.append(_snapshot)
        if len(_trail) > 3:
            _trail.pop(0)
    st.session_state["live_bg_trail"] = _trail
    # Pass all frames except the last (current frame is rendered separately as all_sats)
    _ghost_frames = _trail[:-1] if len(_trail) > 1 else []

    render_map(
        pairs_df,
        selected=live_payload["selected"],
        routes=(live_result or {}).get("routes", []),
        corridor=live_payload["corridor"],
        all_sats=live_bg if show_all_sats else None,
        show_all=show_all_sats,
        height=520,
        trail_frames=_ghost_frames if show_all_sats else [],
    )

    if live_result:
        lm1, lm2, lm3, lm4 = st.columns(4)
        lm1.metric("Live latency",  f"{live_result['total_latency']:.6f} s")
        lm2.metric("Congestion",    f"{live_result['congestion']:.4f}")
        lm3.metric("Feasible",      f"{live_result['repaired_feasibility']['p_feasible']:.0%}")
        lm4.metric("Objective",     f"{live_result['repaired_obj']:.4f}")

        live_route_rows = [
            {
                "Pair":  int(r["m"]) + 1,
                "Route": f"DP{int(r['m'])+1} → L1-{r['layer1']} → L2-{r['layer2']-2} → DT{int(r['m'])+1}",
            }
            for r in live_result["routes"]
        ]
        if live_route_rows:
            st.dataframe(pd.DataFrame(live_route_rows), use_container_width=True, hide_index=True)

    # Auto-refresh every 0.5 s for smoother animation
    time.sleep(0.5)
    st.rerun()
