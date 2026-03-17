from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelBundle:
    crop: str
    pipe: Pipeline
    latest_date: datetime


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["Date"])
    out = df.copy()
    out["day_of_year"] = d.dt.dayofyear
    out["month"] = d.dt.month
    out["t"] = (d - d.min()).dt.days
    return out


def train_for_crop(df: pd.DataFrame, crop: str) -> ModelBundle:
    sub = df[df["Crop"].str.lower() == crop.lower()].copy()
    if sub.empty:
        raise ValueError(f"Unknown crop: {crop}")
    sub = _add_time_features(sub)

    X = sub[["District", "day_of_year", "month", "t"]]
    y = sub["Price"].astype(float).values

    pre = ColumnTransformer(
        transformers=[
            ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
            ("num", "passthrough", ["day_of_year", "month", "t"]),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("model", Ridge(alpha=1.0, random_state=7)),
        ]
    )
    pipe.fit(X, y)
    latest = pd.to_datetime(sub["Date"]).max()
    return ModelBundle(crop=crop, pipe=pipe, latest_date=latest.to_pydatetime())


def predict_next(
    bundle: ModelBundle,
    district: str,
    horizon_days: int = 7,
) -> list[dict]:
    start = bundle.latest_date + timedelta(days=1)
    rows = []
    for i in range(horizon_days):
        d = start + timedelta(days=i)
        rows.append(
            {
                "District": district,
                "Date": d.date().isoformat(),
                "day_of_year": int(d.timetuple().tm_yday),
                "month": int(d.month),
                "t": int((d - bundle.latest_date).days),
            }
        )
    X = pd.DataFrame(rows)[["District", "day_of_year", "month", "t"]]
    yhat = bundle.pipe.predict(X)
    out = []
    for r, p in zip(rows, yhat):
        out.append({"date": r["Date"], "predicted_price": float(round(max(2.0, p), 2))})
    return out


def build_history(df: pd.DataFrame, crop: str, district: str, days: int = 30) -> list[dict]:
    sub = df[
        (df["Crop"].str.lower() == crop.lower())
        & (df["District"].str.lower() == district.lower())
    ].copy()
    if sub.empty:
        return []
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub.sort_values("Date").tail(days)
    return [
        {"date": r["Date"].date().isoformat(), "price": float(round(float(r["Price"]), 2))}
        for _, r in sub.iterrows()
    ]


def demand_levels(latest_by_district: pd.DataFrame) -> pd.DataFrame:
    # demand proxy: rank price within crop (higher price = higher demand)
    if latest_by_district.empty:
        return latest_by_district.assign(demand_level="Low")
    q1 = latest_by_district["Price"].quantile(0.33)
    q2 = latest_by_district["Price"].quantile(0.66)

    def level(p: float) -> str:
        if p >= q2:
            return "High"
        if p >= q1:
            return "Medium"
        return "Low"

    out = latest_by_district.copy()
    out["demand_level"] = out["Price"].astype(float).apply(level)
    return out

