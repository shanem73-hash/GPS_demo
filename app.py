from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from pathlib import Path
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

R_EARTH_KM = 6371.0
GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
LOCAL_TEXTURE_PATH = Path(__file__).parent / "assets" / "earth_beauty.jpg"
EARTH_TEXTURE_URLS = [
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png",
    "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg",
]


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


@st.cache_data(ttl=86400)
def fetch_earth_texture(urls, width: int = 2048, height: int = 1024):
    # Prefer bundled local texture (most reliable for cloud deployments)
    if LOCAL_TEXTURE_PATH.exists():
        img = Image.open(LOCAL_TEXTURE_PATH).convert("RGB")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(img), f"local:{LOCAL_TEXTURE_PATH.name}"

    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            return np.array(img), url
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load Earth texture from any source: {last_err}")


def earth_mesh_with_texture(radius_km: float, texture: np.ndarray, n_lon=180, n_lat=90, opacity=0.75):
    lons = np.linspace(-np.pi, np.pi, n_lon)
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    x = radius_km * np.cos(lat_grid) * np.cos(lon_grid)
    y = radius_km * np.cos(lat_grid) * np.sin(lon_grid)
    z = radius_km * np.sin(lat_grid)

    xv, yv, zv = x.ravel(), y.ravel(), z.ravel()

    tex_h, tex_w, _ = texture.shape
    u = (lon_grid + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - lat_grid) / np.pi
    px = np.clip((u * (tex_w - 1)).astype(int), 0, tex_w - 1)
    py = np.clip((v * (tex_h - 1)).astype(int), 0, tex_h - 1)
    sampled = texture[py, px].reshape(-1, 3)
    vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in sampled]

    I, J, K = [], [], []
    for r in range(n_lat - 1):
        for c in range(n_lon - 1):
            p0 = r * n_lon + c
            p1 = p0 + 1
            p2 = p0 + n_lon
            p3 = p2 + 1
            I.extend([p0, p0])
            J.extend([p2, p1])
            K.extend([p1, p3])

    return go.Mesh3d(
        x=xv,
        y=yv,
        z=zv,
        i=I,
        j=J,
        k=K,
        vertexcolor=vertexcolor,
        opacity=opacity,
        name="Earth (textured)",
        hoverinfo="skip",
        flatshading=False,
    )


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


def snapshot_positions_ecef(satellites, ts, observer=None):
    t = ts.now()
    names, pos, altitudes = [], [], []

    for sat in satellites:
        geocentric = sat.at(t)
        x_km, y_km, z_km = geocentric.frame_xyz(itrs).km  # Earth-fixed frame

        alt_deg = None
        if observer is not None:
            topocentric = (sat - observer).at(t)
            alt, _, _ = topocentric.altaz()
            alt_deg = float(alt.degrees)

        names.append(sat.name)
        pos.append((float(x_km), float(y_km), float(z_km)))
        altitudes.append(alt_deg)

    return names, np.array(pos), altitudes


def orbit_trails_ecef(satellites, ts, minutes_span=45, step_min=5):
    t0 = ts.now()
    mins = np.arange(-minutes_span, minutes_span + step_min, step_min)
    tt = t0.tt + mins / 1440.0
    t_arr = ts.tt_jd(tt)

    trails = {}
    for sat in satellites:
        p = sat.at(t_arr).frame_xyz(itrs).km  # same Earth-fixed frame
        trails[sat.name] = p.T  # (N,3)
    return trails


def make_figure(names, positions, labels=False, trails=None, earth_trace=None, height=920):
    fig = go.Figure()

    if earth_trace is not None:
        fig.add_trace(earth_trace)
    else:
        # Fallback Earth sphere so app still works if textures are unavailable
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 80)
        x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                showscale=False,
                opacity=0.72,
                colorscale=[[0, "#0b3d91"], [1, "#1f77b4"]],
                name="Earth (fallback)",
                hoverinfo="skip",
            )
        )

    # Atmosphere shell for nicer visuals
    u2 = np.linspace(0, 2 * np.pi, 64)
    v2 = np.linspace(0, np.pi, 64)
    r_atm = R_EARTH_KM * 1.025
    xa = r_atm * np.outer(np.cos(u2), np.sin(v2))
    ya = r_atm * np.outer(np.sin(u2), np.sin(v2))
    za = r_atm * np.outer(np.ones_like(u2), np.cos(v2))
    fig.add_trace(
        go.Surface(
            x=xa,
            y=ya,
            z=za,
            showscale=False,
            opacity=0.10,
            colorscale=[[0, "#6bbcff"], [1, "#6bbcff"]],
            hoverinfo="skip",
            name="Atmosphere",
        )
    )

    mode = "markers+text" if labels else "markers"
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode=mode,
            marker=dict(size=5, color="#ffd54f", line=dict(color="#fff8e1", width=0.5), opacity=0.95),
            text=names if labels else None,
            textposition="top center",
            hovertemplate="%{text}<br>x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.0f} km<extra></extra>",
            name="GPS satellites",
        )
    )

    if trails:
        palette = px.colors.qualitative.Set3 + px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly
        for idx, (sat_name, pts) in enumerate(trails.items()):
            clr = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(width=3, color=clr),
                    name=f"Trail: {sat_name}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    lim = 30000
    fig.update_layout(
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            xaxis=dict(range=[-lim, lim], showbackground=False),
            yaxis=dict(range=[-lim, lim], showbackground=False),
            zaxis=dict(range=[-lim, lim], showbackground=False),
            aspectmode="cube",
            bgcolor="#020611",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
        ),
        paper_bgcolor="#020611",
        plot_bgcolor="#020611",
        font=dict(color="#e8f1ff"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
        title="GPS Constellation (Earth-fixed frame, real scale)",
    )
    return fig


def main():
    st.set_page_config(page_title="GPS Demo", page_icon="🛰️", layout="wide")
    st.title("🛰️ GPS Demo: 3D Earth + Live GPS Satellites")
    st.caption("Near real-time = live GPS TLE + current propagation (SGP4). Coordinates shown in Earth-fixed (ITRS/ECEF) frame.")

    with st.sidebar:
        st.markdown("### Display Options")
        labels = st.checkbox("Show satellite labels", value=False)
        show_trails = st.checkbox("Show orbit trails", value=True)
        visible_only = st.checkbox("Only satellites above horizon", value=False)
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=True)
        earth_opacity = st.slider("Earth transparency", min_value=0.15, max_value=1.0, value=0.62, step=0.01)
        view_height = st.slider("3D view height", min_value=760, max_value=1300, value=980, step=20)

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
        names, positions, altitudes = snapshot_positions_ecef(satellites, ts, observer=observer)
    except Exception as e:
        st.error(f"Failed to load GPS satellite data: {e}")
        return

    earth_trace = None
    texture_source = None
    try:
        texture, texture_source = fetch_earth_texture(EARTH_TEXTURE_URLS)
        earth_trace = earth_mesh_with_texture(R_EARTH_KM, texture, opacity=earth_opacity)
    except Exception as e:
        st.warning(f"Earth texture unavailable, using fallback sphere. Details: {e}")

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
        trails = orbit_trails_ecef(selected, ts)

    st.success(f"Showing {len(names)} satellites{' above horizon' if visible_only else ''}")
    st.plotly_chart(
        make_figure(names, positions, labels=labels, trails=trails, earth_trace=earth_trace, height=view_height),
        width="stretch",
        config={"scrollZoom": False},
    )
    if texture_source:
        st.caption(f"Earth texture source: {texture_source}")

    if visible_only:
        st.caption("Visibility criterion: elevation angle > 0° from observer location.")

    if auto_refresh:
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
