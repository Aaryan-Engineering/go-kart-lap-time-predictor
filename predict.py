from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from features import add_pace_history

MODEL = Path("models/prelap_lap_time_model.joblib")
MASTER = Path("data/kart_master_dataset.csv")

PACE_HISTORY_FIELDS = [
    "prev_lap_time_s", "rolling_3_lap_time_s", "session_best_lap_s",
    "prev_sector_1_s", "prev_sector_2_s", "prev_sector_3_s",
]


def num_input(label):
    x = input(f"{label} (Enter=unknown): ").strip()
    if not x:
        return np.nan
    try:
        return float(x)
    except ValueError:
        print("  Couldn't parse that as a number — leaving unknown.")
        return np.nan


def autofill_pace_history(session_id: str) -> dict:
    """
    All six pace-history fields are derivable from the master dataset for
    an in-progress session — no reason to make you type in your own last
    lap time, rolling average, best lap, and three sector splits by hand.
    Returns a dict with whatever's available; missing ones are just absent
    (caller falls back to manual entry for those).
    """
    if not MASTER.exists():
        return {}
    df = add_pace_history(pd.read_csv(MASTER))
    session_laps = df[df["session_id"].astype(str) == session_id]
    if session_laps.empty:
        return {}
    session_laps = session_laps.sort_values("lap_number")

    # the NEXT lap's pace-history would be computed from all laps so far in
    # this session, i.e. treat "one past the last lap number" as the target
    last_lap_num = int(session_laps["lap_number"].max())
    timed = session_laps[session_laps["lap_time_s"].notna()]
    if timed.empty:
        return {}

    result = {
        "prev_lap_time_s": float(timed["lap_time_s"].iloc[-1]),
        "rolling_3_lap_time_s": float(timed["lap_time_s"].tail(3).mean()),
        "session_best_lap_s": float(timed["lap_time_s"].min()),
    }
    for col in ["sector_1_s", "sector_2_s", "sector_3_s"]:
        prev_col = f"prev_{col}"
        val = pd.to_numeric(timed[col], errors="coerce").iloc[-1] if col in timed.columns else np.nan
        if pd.notna(val):
            result[prev_col] = float(val)
    return result


def main():
    if not MODEL.exists():
        print("No model found. Run build_dataset.py, update_lap_times.py, then train_prelap_model.py first.")
        return

    a = joblib.load(MODEL)
    row = {}

    session_id = input("Session ID (Enter to skip pace-history autofill): ").strip()
    autofill = autofill_pace_history(session_id) if session_id else {}
    if session_id and not autofill:
        print(f"  No timed laps found yet for session '{session_id}' — pace-history fields "
              f"will be asked for manually below.")

    for f in a["features"]:
        if f == "track":
            row[f] = input("Track: ").strip()
        elif f in autofill:
            print(f"{f}: using {autofill[f]:.3f} from session {session_id}'s history so far")
            row[f] = autofill[f]
        else:
            row[f] = num_input(f)

    pred = float(a["pipeline"].predict(pd.DataFrame([row]))[0])
    print(f"\nPREDICTED NEXT LAP: {pred:.3f} s")
    if "validation" in a:
        v = a["validation"]
        print(f"Model training laps: {a['training_rows']} | holdout MAE {v['mae_s']:.3f}s")


if __name__ == "__main__":
    main()
