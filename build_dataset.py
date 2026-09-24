from pathlib import Path
from datetime import datetime,timedelta
import pandas as pd
RAW=Path("data/raw"); MASTER=Path("data/kart_master_dataset.csv")
SPEED_SCALE=.1; G_SCALE=.001
COLS=["session_id","date","track","session_start_time","lap_number","lap_timestamp","lap_time_s","sector_1_s","sector_2_s","sector_3_s","max_rpm","max_speed_kmh","avg_speed_kmh","max_lateral_g","max_longitudinal_g","temperature_c","humidity_pct","pressure_hpa","wind_speed_kmh","wind_direction_deg","wind_gust_kmh","rainfall_mm","rf_pressure_cold","rr_pressure_cold","lf_pressure_cold","lr_pressure_cold","front_sprocket","rear_sprocket","notes"]
def main():
    sid=input("Session ID: ").strip(); date=input("Date (YYYY-MM-DD): ").strip(); track=input("Track: ").strip(); start=input("Session start time (HH:MM:SS): ").strip()
    files=sorted((RAW/sid).glob("*.csv"))
    if not files: print("No CSV files found."); return
    try:t0=datetime.strptime(f"{date} {start}","%Y-%m-%d %H:%M:%S")
    except:t0=None
    rows=[]; elapsed=0
    for n,p in enumerate(files,1):
        d=pd.read_csv(p); c={str(x).strip().lower():x for x in d.columns}
        def s(name): return pd.to_numeric(d[c[name.lower()]],errors="coerce") if name.lower() in c else pd.Series(dtype=float)
        rpm=s("RPM"); speed=s("Speed GPS")*.1; lat=s("Gf. X")*.001; lon=s("Gf. Y")*.001
        ts=(t0+timedelta(seconds=elapsed)).isoformat(timespec="seconds") if t0 else ""
        rows.append(dict(session_id=sid,date=date,track=track,session_start_time=start,lap_number=n,lap_timestamp=ts,lap_time_s="",sector_1_s="",sector_2_s="",sector_3_s="",max_rpm=rpm.max(),max_speed_kmh=speed.max(),avg_speed_kmh=speed.mean(),max_lateral_g=lat.abs().max(),max_longitudinal_g=lon.abs().max(),temperature_c="",humidity_pct="",pressure_hpa="",wind_speed_kmh="",wind_direction_deg="",wind_gust_kmh="",rainfall_mm="",rf_pressure_cold="",rr_pressure_cold="",lf_pressure_cold="",lr_pressure_cold="",front_sprocket="",rear_sprocket="",notes=""))
    old=pd.read_csv(MASTER) if MASTER.exists() else pd.DataFrame(columns=COLS)
    if not old.empty and (old.session_id.astype(str)==sid).any(): print("Session already exists; use a new Session ID."); return
    # keep every column the existing file has (e.g. hot tyre pressures), not
    # just this script's own COLS list — selecting [COLS] on old+new combined
    # would silently drop any column old rows have that this script doesn't
    # know about
    all_cols=list(dict.fromkeys(list(old.columns)+COLS))
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True)[all_cols].to_csv(MASTER,index=False)
    print(f"Added {len(rows)} laps. Alfano scaling fixed: speed x0.1, G x0.001.")
if __name__=="__main__": main()
