# GPS_demo

3D Earth visualization with near real-time GPS satellite positions.

## What this does
- Pulls current GPS operational TLE data from CelesTrak
- Propagates satellite position at current time using Skyfield/SGP4
- Plots Earth + GPS satellites in an interactive 3D model (Plotly)

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- This is near real-time via live TLE + propagation, not direct telemetry.
- Data source: https://celestrak.org/
