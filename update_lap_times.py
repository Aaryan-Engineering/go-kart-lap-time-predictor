from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

M = Path("data/kart_master_dataset.csv")


def ask_positive(prompt):
    while True:
        x = input(prompt).strip()
        if not x:
            return None
        try:
            v = float(x)
            if v > 0:
                return v
        except ValueError:
            pass
        print("  Enter a positive number, or leave blank to skip.")


def main():
    df = pd.read_csv(M)
    # when a column is entirely blank, pandas infers it as float64 (all-NaN)
    # rather than string — assigning a timestamp string into it then raises
    # a dtype error. Force it to string dtype up front so writes always work.
    if "lap_timestamp" not in df.columns:
        df["lap_timestamp"] = ""
    df["lap_timestamp"] = df["lap_timestamp"].astype(object)

    mask = df.lap_time_s.isna() | (df.lap_time_s.astype(str).str.strip() == "")
    for i, r in df[mask].iterrows():
        v = ask_positive(f"{r.session_id} lap {r.lap_number} actual lap time (s), Enter=skip: ")
        if v is not None:
            df.at[i, "lap_time_s"] = v

    # sector times — only prompted for laps that now have a lap_time_s but
    # are still missing one or more sectors, so re-running this script
    # doesn't re-ask for sectors you already entered
    sector_cols = ["sector_1_s", "sector_2_s", "sector_3_s"]
    for c in sector_cols:
        if c not in df.columns:
            df[c] = ""
    has_time = df.lap_time_s.notna() & (df.lap_time_s.astype(str).str.strip() != "")
    missing_sector = df[sector_cols].isna().any(axis=1) | (df[sector_cols].astype(str) == "").any(axis=1)
    for i, r in df[has_time & missing_sector].iterrows():
        print(f"{r.session_id} lap {r.lap_number} (lap time {r.lap_time_s}s) — sector times (Enter=skip any):")
        for c, label in zip(sector_cols, ["sector 1", "sector 2", "sector 3"]):
            existing = df.at[i, c]
            if pd.notna(existing) and str(existing).strip():
                continue  # already have this one
            v = ask_positive(f"  {label} (s): ")
            if v is not None:
                df.at[i, c] = v

    # reconstruct lap_timestamp from session start + cumulative lap time.
    # IMPORTANT: if a lap in the middle of a session has no time, elapsed
    # can't advance correctly past it — every timestamp AFTER that gap is
    # then wrong (too early) but was being written anyway with no warning.
    # Now: stop advancing timestamps for the rest of that session once a
    # gap is hit, and say so, rather than silently writing bad values.
    for sid, g in df.groupby("session_id", sort=False):
        try:
            t = datetime.strptime(f"{g.date.iloc[0]} {g.session_start_time.iloc[0]}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        elapsed = 0
        gap_hit = False
        for i in g.sort_values("lap_number").index:
            if gap_hit:
                df.at[i, "lap_timestamp"] = ""  # can't trust anything after the gap
                continue
            v = pd.to_numeric(df.at[i, "lap_time_s"], errors="coerce")
            if pd.isna(v):
                print(f"  NOTE: {sid} lap {df.at[i,'lap_number']} still has no lap_time_s — "
                      f"can't reconstruct timestamps for the rest of this session (every lap "
                      f"after this one depends on knowing how long this one took). Fill it in "
                      f"and re-run to get those timestamps back.")
                gap_hit = True
                df.at[i, "lap_timestamp"] = ""
                continue
            df.at[i, "lap_timestamp"] = (t + timedelta(seconds=elapsed)).isoformat(timespec="seconds")
            elapsed += float(v)

    df.to_csv(M, index=False)
    print("Updated master dataset.")


if __name__ == "__main__":
    main()
