# KartiQ / Kart Lap-Time Predictor v4

## What is fixed
- Correct Alfano scaling: Speed GPS ×0.1 and Gf.X/Gf.Y ×0.001.
- Separate true pre-lap model: no max speed/RPM/G telemetry outcomes.
- Pace-history features: previous lap, previous 3-lap average, session best and previous sectors.
- Validation split by session to reduce leakage between laps from the same session.
- Four-wheel cold tyre pressures.
- Dashboard aligned with the new model.
- Lap timestamps are rebuilt from session start and known lap times.

## Run
```bash
python3 -m pip install -r requirements.txt
python3 build_dataset.py
python3 update_lap_times.py
python3 train_prelap_model.py
python3 predict.py
python3 build_dashboard.py
```

Add each new session under `data/raw/S002/`, `S003/`, etc. The model is retrained from
the accumulated labelled dataset.

## Important
The first lap of a session has no previous-lap information, so those fields are naturally
unknown. Weather should eventually be automatically attached from BOM using lap timestamps.
The Alfano export itself does not provide a reliable absolute timestamp per lap.
