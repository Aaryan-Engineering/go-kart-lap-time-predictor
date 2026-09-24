from pathlib import Path
import webbrowser
import joblib
import pandas as pd
import plotly.graph_objects as go

from features import WEATHER_SETUP_FEATURES, add_pace_history

M = Path("data/kart_master_dataset.csv")
MODEL = Path("models/prelap_lap_time_model.joblib")
OUT = Path("dashboard.html")

PACE_HISTORY_FIELDS = [
    "prev_lap_time_s", "rolling_3_lap_time_s", "session_best_lap_s",
    "prev_sector_1_s", "prev_sector_2_s", "prev_sector_3_s",
]


def fig_feature_importance(model_bundle):
    """
    SimpleImputer silently drops any column that was all-NaN in training
    (can't compute a median for it), so the transformed feature count can
    be smaller than len(features). Filter using the imputer's own
    statistics_ so names line up with what the model actually saw, rather
    than crashing or mislabeling bars.
    """
    pipe = model_bundle["pipeline"]
    features = model_bundle["features"]
    num_features = [f for f in features if f != "track"]
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["imp"]
    kept_mask = ~pd.isna(imputer.statistics_)
    num_names = [f for f, keep in zip(num_features, kept_mask) if keep]
    ohe_names = list(pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"].get_feature_names_out(["track"]))
    all_names = num_names + ohe_names

    importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=all_names).sort_values()
    fig = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(title="Pre-lap model — feature importance", xaxis_title="Importance",
                       height=max(350, 28 * len(importances)))
    return fig


def build_notes(d, model_bundle):
    notes = []
    n = len(d)
    notes.append(f"<li><b>{n} labeled laps</b> across {d['date'].nunique()} sessions/days. "
                  f"10 is the training minimum, 30+ is what the README says actually matters — "
                  f"treat every chart here as directional until there's more data.</li>")

    if "validation" in model_bundle:
        v = model_bundle["validation"]
        r2_str = f"{v['r2']:.3f}" if v["r2"] == v["r2"] else "n/a (test session had 1 lap)"
        notes.append(f"<li>Holdout validation (split by session): MAE {v['mae_s']:.3f}s, "
                      f"R\u00b2 {r2_str}. With this few sessions, a single held-out session can "
                      f"swing this a lot — don't read a bad R\u00b2 here as \"the model is "
                      f"broken\", it's what small-sample validation looks like.</li>")

    zero_var = [c for c in WEATHER_SETUP_FEATURES if c in d.columns and pd.to_numeric(d[c], errors="coerce").nunique(dropna=True) <= 1]
    if zero_var:
        notes.append(f"<li>These features have no variation across your labeled laps yet, so "
                      f"the model can't learn anything from them: {zero_var}</li>")

    empty_pace = [c for c in PACE_HISTORY_FIELDS if c in d.columns and d[c].isna().all()]
    if empty_pace:
        notes.append(f"<li><b>These pace-history features are entirely empty</b> and "
                      f"contributed nothing to this model: {empty_pace}. Sector-time fields "
                      f"need sector_1_s/2_s/3_s filled in via update_lap_times.py; the others "
                      f"need multiple timed laps within a single session.</li>")

    return f"<ul>{''.join(notes)}</ul>"


def main():
    if not M.exists() or not MODEL.exists():
        print("Run train_prelap_model.py first.")
        return

    d = add_pace_history(pd.read_csv(M))
    d = d[d.lap_time_s.notna()].copy()
    for c in ["lap_time_s", "temperature_c", "wind_speed_kmh",
              "rf_pressure_cold", "rr_pressure_cold", "lf_pressure_cold", "lr_pressure_cold"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    model_bundle = joblib.load(MODEL)

    figs = []
    figs.append(("Lap Times", go.Figure(go.Scatter(x=d.date, y=d.lap_time_s, mode="markers+lines"))))
    for col, title in [("temperature_c", "Temperature"), ("wind_speed_kmh", "Wind Speed")]:
        x = d.dropna(subset=[col])
        figs.append((title, go.Figure(go.Scatter(x=x[col], y=x.lap_time_s, mode="markers"))))
    d["avg_cold_psi"] = d[["rf_pressure_cold", "rr_pressure_cold", "lf_pressure_cold", "lr_pressure_cold"]].mean(axis=1)
    x = d.dropna(subset=["avg_cold_psi"])
    figs.append(("Cold Tyre Pressure", go.Figure(go.Scatter(x=x.avg_cold_psi, y=x.lap_time_s, mode="markers"))))
    figs.append(("Feature Importance", fig_feature_importance(model_bundle)))

    body = ""
    for i, (t, f) in enumerate(figs):
        body += f"<section><h2>{t}</h2>{f.to_html(full_html=False, include_plotlyjs='inline' if i == 0 else False)}</section>"

    v = model_bundle["validation"]
    r2_str = f"{v['r2']:.3f}" if v["r2"] == v["r2"] else "n/a"
    body += (f"<section><h2>Model</h2><p>{len(d)} labelled laps | validation MAE {v['mae_s']:.3f}s | "
             f"RMSE {v['rmse_s']:.3f}s | R\u00b2 {r2_str}</p>"
             f"<p>Validation is split by session, not individual laps.</p></section>")
    body += f"<section><h2>Notes & Caveats</h2>{build_notes(d, model_bundle)}</section>"

    OUT.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>KartiQ Dashboard</title>"
        "<style>body{font-family:Arial;background:#f5f5f5}section{background:white;max-width:1100px;"
        "margin:20px auto;padding:20px;border-radius:10px}li{margin-bottom:8px;font-size:14px}</style>"
        f"</head><body><header><h1>\U0001F3CE KartiQ Lap-Time Dashboard</h1></header>{body}</body></html>",
        encoding="utf-8",
    )
    webbrowser.open(OUT.resolve().as_uri())
    print(OUT.resolve())


if __name__ == "__main__":
    main()
