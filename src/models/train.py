import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

FEATURES = ["avg_tone", "event_count", "tone_dispersion"]

def _wide_to_long(df: pd.DataFrame, currencies: List[str] = None) -> pd.DataFrame:
    # ensure date column exists
    df2 = df.reset_index()  # preserves date as column if index is DatetimeIndex or MultiIndex
    if currencies is None:
        tone_cols = [c for c in df2.columns if c.startswith("avg_tone_")]
        currencies = sorted([c.replace("avg_tone_", "") for c in tone_cols])
    rows = []
    for c in currencies:
        avg_col = f"avg_tone_{c}"
        ev_col = f"event_count_{c}"
        disp_col = f"tone_dispersion_{c}"
        # return candidate names: uppercase currency ticker or lowercase
        ret_col = None
        for cand in (c.upper(), c.lower(), f"{c}_ret"):
            if cand in df2.columns:
                ret_col = cand
                break
        sub = df2[[col for col in [df2.columns[0], avg_col, ev_col, disp_col] if col in df2.columns]].copy()
        sub = sub.rename(columns={df2.columns[0]: "date",
                                  avg_col: "avg_tone",
                                  ev_col: "event_count",
                                  disp_col: "tone_dispersion"})
        if ret_col:
            sub["next_day_return"] = df2[ret_col].values
        else:
            # fallback to any 'next_day_return' column if present
            if "next_day_return" in df2.columns:
                sub["next_day_return"] = df2["next_day_return"].values
            else:
                sub["next_day_return"] = pd.NA
        sub["currency"] = c
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    # ensure feature columns exist
    for f in FEATURES:
        if f not in long.columns:
            long[f] = 0
    return long

def train_and_save_model(df: pd.DataFrame) -> Tuple[object, object]:
    """
    Accepts wide-format df (avg_tone_<ccy>, event_count_<ccy>, tone_dispersion_<ccy>, returns like TRY/HUF/...)
    Converts to long format and trains a model to predict next-day return per currency.
    """
    # build currency list from avg_tone_ columns
    tone_cols = [c for c in df.columns if c.startswith("avg_tone_")]
    if not tone_cols:
        raise ValueError("No avg_tone_ columns found in df")
    currencies = sorted([c.replace("avg_tone_", "") for c in tone_cols])

    long = _wide_to_long(df, currencies=currencies).dropna(subset=["next_day_return"])
    X = long[FEATURES].fillna(0)
    y = long["next_day_return"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X_scaled, y)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/xgb_sentiment.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model trained & saved")
    return model, scaler