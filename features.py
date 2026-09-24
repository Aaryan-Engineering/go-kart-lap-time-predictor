import pandas as pd
WEATHER_SETUP_FEATURES=["temperature_c","humidity_pct","pressure_hpa","wind_speed_kmh","wind_direction_deg","wind_gust_kmh","rainfall_mm","rf_pressure_cold","rr_pressure_cold","lf_pressure_cold","lr_pressure_cold","front_sprocket","rear_sprocket"]
PRELAP_FEATURES=["track",*WEATHER_SETUP_FEATURES,"prev_lap_time_s","rolling_3_lap_time_s","session_best_lap_s","prev_sector_1_s","prev_sector_2_s","prev_sector_3_s"]
def add_pace_history(df):
    d=df.copy().sort_values(["session_id","lap_number"])
    d["lap_time_s"]=pd.to_numeric(d["lap_time_s"],errors="coerce")
    g=d.groupby("session_id",sort=False)
    d["prev_lap_time_s"]=g["lap_time_s"].shift(1)
    d["rolling_3_lap_time_s"]=g["lap_time_s"].transform(lambda s:s.shift(1).rolling(3,min_periods=1).mean())
    d["session_best_lap_s"]=g["lap_time_s"].transform(lambda s:s.shift(1).cummin())
    for s in ["sector_1_s","sector_2_s","sector_3_s"]: d[s]=pd.to_numeric(d[s],errors="coerce")
    d["prev_sector_1_s"]=g["sector_1_s"].shift(1); d["prev_sector_2_s"]=g["sector_2_s"].shift(1); d["prev_sector_3_s"]=g["sector_3_s"].shift(1)
    return d
