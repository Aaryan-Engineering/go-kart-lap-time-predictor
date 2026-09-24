# Kart Lap-Time Predictor

A machine learning pipeline that predicts go-kart lap times from weather conditions, tyre
pressure, and driver pace history — built from real session telemetry and race-day data
collected at the Go Kart Club of Victoria.

Personal engineering project exploring the same class of problem motorsport performance
teams work on: how track conditions and setup translate into lap time, and how to build
that pipeline with disciplined, leakage-free machine learning practice.

## What it does

- **Ingests raw telemetry**: parses GPS/IMU data from an Alfano6 lap logger (speed, RPM,
  lateral/longitudinal G) with correct unit scaling
- **Combines it with real conditions data**: session weather (temperature, humidity,
  pressure, wind speed/direction/gust) and four-corner tyre pressure, both cold and hot
- **Trains a genuine pre-lap forecasting model**: a Random Forest regressor that predicts
  lap time using only information available *before* the lap starts — track, weather,
  tyre pressure, and pace history (previous lap time, rolling 3-lap average, session-best
  time, previous sector splits)
- **Deliberately excludes post-lap telemetry** (max speed, max RPM, peak G) from the
  forecasting model, since those are only known once the lap is already over — including
  them would let the model "predict" using information that doesn't exist yet at
  prediction time. This distinction between pre-lap and post-lap features is the
  central design decision in the project (see [How it works](#how-it-works))
- **Validates without leakage**: train/test splits are grouped by session
  (`GroupShuffleSplit`), so laps from the same session can never appear on both sides of
  the split
- **Visualises results**: a self-contained, interactive HTML dashboard (Plotly) showing
  lap time vs. temperature, wind speed, and tyre pressure, plus feature importance and
  automatically generated data-quality warnings (small sample size, zero-variance
  features, missing pace history)

## How it works

The forecasting model is a **Random Forest Regressor** (scikit-learn), trained as a
supervised regression problem: each training example is one real, timed lap, with lap
time (seconds) as the label.

1. **Imputation** — `SimpleImputer` fills missing values (median for numeric features)
   so one missing reading doesn't discard an entire lap
2. **Encoding** — `OneHotEncoder` converts the categorical `track` feature into a form
   the model can use
3. **Training** — the forest fits on ~80% of labelled laps
4. **Validation** — evaluated on the held-out ~20%, split by session rather than by
   individual lap, specifically to stop the model exploiting session-specific quirks as
   a shortcut instead of learning genuine patterns
5. **Feature importance** — reports which inputs the model relied on most, useful for
   sanity-checking that it's learning something sensible rather than noise

A Random Forest was chosen over a linear model because the relationship between
conditions and lap time is almost certainly non-linear (e.g. grip probably doesn't
degrade linearly with temperature), and the dataset is small enough that a
tree-ensemble's ability to model non-linear interactions without heavy feature
engineering is a good fit.

## Project structure

```
build_dataset.py        # ingest raw Alfano CSVs into the master dataset
update_lap_times.py     # add real lap times / sector times, reconstruct timestamps
train_prelap_model.py   # train the forecasting model (pre-lap features only)
predict.py              # predict a lap time, with pace-history autofill from past sessions
build_dashboard.py      # generate the interactive HTML dashboard
features.py             # shared feature definitions (pre-lap vs. telemetry)
data/
  kart_master_dataset.csv   # accumulated, labelled dataset — one row per lap
  raw/<session_id>/         # raw Alfano lap CSVs per session
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python3 build_dataset.py       # ingest a new session's raw lap CSVs
python3 update_lap_times.py    # enter real lap times (and sector times, if available)
python3 train_prelap_model.py  # train the model on all labelled laps so far
python3 predict.py             # predict a lap time for given/known conditions
python3 build_dashboard.py     # generate and open the HTML dashboard
```

Add each new session's raw CSVs under `data/raw/<session_id>/`, then run the four steps
above in order. The dataset and model both grow additively — no need to retrain from
scratch each time.

## Known limitations

- **Small sample size**: the model currently trains on a low double-digit number of
  labelled laps. 30+ is a more meaningful starting point; the dashboard's own warnings
  flag this explicitly rather than presenting results as more reliable than they are.
- **Pace-history cold start**: features like previous-lap time have nothing to reference
  on a session's first lap, or in a dataset where each session only has one timed lap.
- **Weather is manually entered**, not yet automatically matched to session timestamps
  (the Alfano export doesn't provide a reliable per-lap absolute timestamp, which is a
  prerequisite for automating that match reliably).
- **Chassis setup is not yet modelled** — see Roadmap.

## Roadmap

- **Chassis setup parameters**: rear axle track width, camber, caster, and ride height —
  these interact with grip and cornering behaviour in ways a point-mass or purely
  statistical model doesn't capture well, and will need either careful feature
  engineering or a physics-informed approach alongside the ML model
- **Gear ratio (front/rear sprocket size)**: incorporate the accel/top-speed trade-off
  from drivetrain gearing into the feature set
- **Automated weather matching**: reliable per-lap timestamps, then automatic
  station-based historical weather lookup instead of manual entry
- **Sector-level prediction**: extend beyond whole-lap time to per-sector forecasts

## Background

Built as a personal engineering project — telemetry pipeline, feature engineering,
model validation, and visualisation, applied to a domain (kart/motorsport performance)
directly relevant to Formula 1 engineering work.
