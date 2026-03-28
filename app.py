from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from skyfield.api import EarthSatellite, load, wgs84

R_EARTH_KM = 6371.0
GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"


def earth_sphere(radius_km: float = R_EARTH_KM, n: int = 60):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


@st.cache_data(ttl=300)
def fetch_gps_tles(url: str):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    sats = []
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            sats.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return sats


def load_satellites(sat_tles, ts):
    return [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in sat_tles]


def snapshot_positions(satellites, ts, observer=None):
    t = ts.now()
    names, pos, altitudes = [], [], []

    for sat in satellites:
        geocentric = sat.at(t)
        x_km, y_km, z_km = geocentric.position.km

        alt_deg = None
        if observer is not None:
            topocentric = (sat - observer).at(t)
            alt, _, _ = topocentric.altaz()
            alt_deg = float(alt.degrees)

        names.append(sat.name)
        pos.append((float(x_km), float(y_km), float(z_km)))
        altitudes.append(alt_deg)

    return names, np.array(pos), altitudes


def orbit_trails(satellites, ts, minutes_span=45, step_min=5):
    # Times around now: [-span, ..., +span] minutes
    t0 = ts.now()
    mins = np.arange(-minutes_span, minutes_span + step_min, step_min)
    tt = t0.tt + mins / 1440.0
    t_arr = ts.tt_jd(tt)

    trails = {}
    for sat in satellites:
        p = sat.at(t_arr).position.km
        trails[sat.name] = p.T  # shape (N,3)
    return trails


def make_figure(names, positions, labels=False, trails=None):
    ex, ey, ez = earth_sphere()

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=ex,
            y=ey,
            z=ez,
            showscale=False,
            opacity=0.85,
            colorscale=[[0, "#0b3d91"], [1, "#1f77b4"]],
            name="Earth",
            hoverinfo="skip",
        )
    )

    mode = "markers+text" if labels else "markers"
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode=mode,
            marker=dict(size=4, color="#ffcc00"),
            text=names if labels else None,
            textposition="top center",
            hovertemplate="%{text}<br>x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.0f} km<extra></extra>",
            name="GPS satellites",
        )
    )

    if trails:
        for sat_name, pts in trails.items():
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(width=2, color="rgba(255, 204, 0, 0.55)"),
                    name=f"Trail: {sat_name}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    lim = 28000
    fig.update_layout(
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            xaxis=dict(range=[-lim, lim]),
            yaxis=dict(range=[-lim, lim]),
            zaxis=dict(range=[-lim, lim]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=760,
        title="GPS Constellation (near real-time from live TLE)",
    )
    return fig


def main():
    st.set_page_config(page_title="GPS Demo", page_icon="🛰️", layout="wide")
    st.title("🛰️ GPS Demo: 3D Earth + Live GPS Satellites")
    st.caption("Satellite positions are propagated from current CelesTrak TLE data (near real-time, not raw telemetry).")

    with st.sidebar:
        st.markdown("### Display Options")
        labels = st.checkbox("Show satellite labels", value=False)
        show_trails = st.checkbox("Show orbit trails", value=True)
        visible_only = st.checkbox("Only satellites above horizon", value=False)
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=True)

        st.markdown("### Observer Location (for horizon filter)")
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=41.8781, step=0.0001)
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-87.6298, step=0.0001)
        elev_m = st.number_input("Elevation (m)", min_value=-100.0, max_value=9000.0, value=180.0, step=1.0)

        trails_limit = st.slider("Max satellites with trails", min_value=3, max_value=31, value=12)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        st.write(f"UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z")

    try:
        ts = load.timescale()
        sat_tles = fetch_gps_tles(GPS_TLE_URL)
        satellites = load_satellites(sat_tles, ts)

        observer = wgs84.latlon(lat, lon, elevation_m=elev_m)
        names, positions, altitudes = snapshot_positions(satellites, ts, observer=observer)
    except Exception as e:
        st.error(f"Failed to load GPS data: {e}")
        return

    # Filter visible only if requested
    if visible_only:
        mask = [alt is not None and alt > 0 for alt in altitudes]
        names = [n for n, m in zip(names, mask) if m]
        positions = np.array([p for p, m in zip(positions, mask) if m]) if any(mask) else np.empty((0, 3))
        sats_for_trails = [s for s, m in zip(satellites, mask) if m]
    else:
        sats_for_trails = satellites

    if len(names) == 0:
        st.warning("No GPS satellites are above horizon for this observer at this moment.")
        return

    trails = None
    if show_trails:
        selected = sats_for_trails[:trails_limit]
        trails = orbit_trails(selected, ts)

    st.success(f"Showing {len(names)} satellites{' above horizon' if visible_only else ''}")
    st.plotly_chart(make_figure(names, positions, labels=labels, trails=trails), use_container_width=True, config={"scrollZoom": False})

    if visible_only:
        st.caption("Visibility test uses observer location and altitude > 0° horizon criterion.")

    st.info("Note: 'Real-time' here means live orbital elements + current-time propagation (SGP4).")

    if auto_refresh:
        # Browser-side refresh every 30 seconds
        components.html(
            """
            <script>
            setTimeout(function() { window.parent.location.reload(); }, 30000);
            </script>
            """,
            height=0,
        )


if __name__ == "__main__":
    main()
