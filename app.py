from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from skyfield.api import EarthSatellite, load

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


def current_positions_ecef(sat_tles):
    ts = load.timescale()
    t = ts.now()
    names, pos = [], []
    for name, l1, l2 in sat_tles:
        sat = EarthSatellite(l1, l2, name, ts)
        geocentric = sat.at(t)
        x_km, y_km, z_km = geocentric.position.km
        names.append(name)
        pos.append((float(x_km), float(y_km), float(z_km)))
    return names, np.array(pos)


def make_figure(names, positions):
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

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(size=4, color="#ffcc00"),
            text=names,
            hovertemplate="%{text}<br>x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.0f} km<extra></extra>",
            name="GPS satellites",
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

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        st.write(f"UTC now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z")

    try:
        sat_tles = fetch_gps_tles(GPS_TLE_URL)
        names, positions = current_positions_ecef(sat_tles)
    except Exception as e:
        st.error(f"Failed to load GPS data: {e}")
        return

    st.success(f"Loaded {len(names)} GPS satellites")
    st.plotly_chart(make_figure(names, positions), use_container_width=True, config={"scrollZoom": False})

    st.info("Note: 'Real-time' here means live orbital elements + current-time propagation (SGP4).")


if __name__ == "__main__":
    main()
