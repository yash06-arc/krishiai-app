from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from data_gen import CROPS, DISTRICTS, ensure_dataset
from ml import build_history, demand_levels, predict_next, train_for_crop


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.abspath(os.path.join(ROOT, "..", "data", "prices_karnataka.csv"))


def load_df() -> pd.DataFrame:
    ensure_dataset(DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    # normalize schema just in case
    df["Crop"] = df["Crop"].astype(str)
    df["District"] = df["District"].astype(str)
    df["Date"] = df["Date"].astype(str)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    return df


@lru_cache(maxsize=1)
def _df_cached() -> pd.DataFrame:
    return load_df()


@lru_cache(maxsize=64)
def _model_for_crop(crop: str):
    return train_for_crop(_df_cached(), crop=crop)


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/meta")
def meta():
    return jsonify({"districts": DISTRICTS, "crops": CROPS})


@app.get("/prices")
def prices():
    crop = (request.args.get("crop") or "").strip()
    district = (request.args.get("district") or "").strip()

    df = _df_cached()
    f = df.copy()
    if crop:
        f = f[f["Crop"].str.lower() == crop.lower()]
    if district:
        f = f[f["District"].str.lower() == district.lower()]

    if f.empty:
        return jsonify({"items": []})

    f["Date"] = pd.to_datetime(f["Date"])
    latest_date = f["Date"].max()
    latest = f[f["Date"] == latest_date].copy()

    # Ensure one row per Crop+District at latest date
    latest = (
        latest.sort_values(["Crop", "District", "Date"])
        .groupby(["Crop", "District"], as_index=False)
        .tail(1)
    )

    out = demand_levels(latest)
    out = out.sort_values(["Crop", "District"])

    items = [
        {
            "crop": r["Crop"],
            "district": r["District"],
            "date": r["Date"].date().isoformat(),
            "price": float(round(float(r["Price"]), 2)),
            "demand_level": r["demand_level"],
        }
        for _, r in out.iterrows()
    ]
    return jsonify({"items": items, "latest_date": latest_date.date().isoformat()})


@app.get("/predict")
def predict():
    crop = (request.args.get("crop") or "").strip()
    district = (request.args.get("district") or "").strip()
    if not crop or not district:
        return jsonify({"error": "Missing crop or district"}), 400

    df = _df_cached()
    bundle = _model_for_crop(crop)

    forecast = predict_next(bundle, district=district, horizon_days=7)
    predicted_price = forecast[0]["predicted_price"] if forecast else None
    history = build_history(df, crop=crop, district=district, days=30)

    return jsonify(
        {
            "crop": crop,
            "district": district,
            "predicted_price": predicted_price,
            "history": history,
            "forecast": forecast,
        }
    )


@app.get("/best-market")
def best_market():
    crop = (request.args.get("crop") or "").strip()
    if not crop:
        return jsonify({"error": "Missing crop"}), 400

    df = _df_cached()
    f = df[df["Crop"].str.lower() == crop.lower()].copy()
    if f.empty:
        return jsonify({"error": "Unknown crop"}), 400

    f["Date"] = pd.to_datetime(f["Date"])
    latest_date = f["Date"].max()
    latest = f[f["Date"] == latest_date].copy()
    latest = (
        latest.sort_values(["District", "Date"])
        .groupby(["District"], as_index=False)
        .tail(1)
        .sort_values("Price", ascending=False)
    )

    district_prices = [
        {"district": r["District"], "price": float(round(float(r["Price"]), 2))}
        for _, r in latest.iterrows()
    ]

    best = district_prices[0] if district_prices else {"district": None, "price": None}
    return jsonify(
        {
            "crop": crop,
            "best_district": best["district"],
            "best_price": best["price"],
            "district_prices": district_prices,
            "latest_date": latest_date.date().isoformat(),
        }
    )


@app.get("/demand")
def demand():
    crop = (request.args.get("crop") or "").strip()
    if not crop:
        return jsonify({"error": "Missing crop"}), 400

    df = _df_cached()
    f = df[df["Crop"].str.lower() == crop.lower()].copy()
    if f.empty:
        return jsonify({"items": []})

    f["Date"] = pd.to_datetime(f["Date"])
    latest_date = f["Date"].max()
    latest = f[f["Date"] == latest_date].copy()
    latest = (
        latest.sort_values(["District", "Date"])
        .groupby(["District"], as_index=False)
        .tail(1)
    )
    out = demand_levels(latest)
    items = [
        {
            "district": r["District"],
            "demand_level": r["demand_level"],
            "price": float(round(float(r["Price"]), 2)),
        }
        for _, r in out.iterrows()
    ]
    return jsonify({"crop": crop, "items": items, "latest_date": latest_date.date().isoformat()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

