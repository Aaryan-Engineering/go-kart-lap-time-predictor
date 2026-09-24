from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from features import PRELAP_FEATURES,add_pace_history
M=Path("data/kart_master_dataset.csv"); OUT=Path("models/prelap_lap_time_model.joblib")
NUM=[x for x in PRELAP_FEATURES if x!="track"]
def main():
    df=add_pace_history(pd.read_csv(M))
    for c in NUM+["lap_time_s"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["lap_time_s"])
    if len(df)<10: print(f"{len(df)} labelled laps; need at least 10, with 30+ much more useful."); return
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=df.session_id))
    tr,te=df.iloc[a],df.iloc[b]
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),NUM),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),["track"])])
    pipe=Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=500,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    pipe.fit(tr[PRELAP_FEATURES],tr.lap_time_s); pred=pipe.predict(te[PRELAP_FEATURES])
    v={"mae_s":float(mean_absolute_error(te.lap_time_s,pred)),"rmse_s":float(np.sqrt(mean_squared_error(te.lap_time_s,pred))),"r2":float(r2_score(te.lap_time_s,pred)) if len(te)>1 else float("nan")}
    pipe.fit(df[PRELAP_FEATURES],df.lap_time_s); OUT.parent.mkdir(exist_ok=True); joblib.dump({"pipeline":pipe,"features":PRELAP_FEATURES,"training_rows":len(df),"validation":v},OUT)
    print(f"Trained on {len(df)} laps | MAE {v['mae_s']:.3f}s | RMSE {v['rmse_s']:.3f}s | R2 {v['r2']:.3f}")
if __name__=="__main__": main()
