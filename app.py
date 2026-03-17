from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, Tuple

import pandas as pd  # type: ignore
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS  # type: ignore

from data_gen import CROPS, DISTRICTS, ensure_dataset  # type: ignore
from ml import build_history, demand_levels, predict_next, train_for_crop  # type: ignore


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.abspath(os.path.join(ROOT, "..", "data", "prices_karnataka.csv"))


KARNATAKA_COORDS: Dict[str, Tuple[float, float]] = {
    "Bangalore Urban": (12.9716, 77.5946),
    "Bangalore Rural": (13.2846, 77.6070),
    "Mysuru": (12.2958, 76.6394),
    "Mandya": (12.5242, 76.8958),
    "Tumakuru": (13.3392, 77.1010),
    "Kolar": (13.1357, 78.1326),
    "Chikkaballapur": (13.4351, 77.7315),
    "Ramanagara": (12.7225, 77.2806),
    "Hassan": (13.0072, 76.0962),
    "Chikkamagaluru": (13.3161, 75.7720),
    "Kodagu": (12.4244, 75.7382),
    "Dakshina Kannada": (12.9141, 74.8560),
    "Udupi": (13.3409, 74.7421),
    "Uttara Kannada": (14.7950, 74.6869),
    "Shivamogga": (13.9299, 75.5681),
    "Davangere": (14.4644, 75.9218),
    "Chitradurga": (14.2266, 76.4000),
    "Ballari": (15.1394, 76.9214),
    "Koppal": (15.3456, 76.1548),
    "Raichur": (16.2076, 77.3463),
    "Kalaburagi": (17.3297, 76.8343),
    "Yadgir": (16.7730, 77.1376),
    "Bidar": (17.9149, 77.5046),
    "Vijayapura": (16.8302, 75.7100),
    "Bagalkot": (16.1850, 75.6961),
    "Belagavi": (15.8497, 74.4977),
    "Dharwad": (15.4589, 75.0078),
    "Gadag": (15.4310, 75.6350),
    "Haveri": (14.7959, 75.4045),
}


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


@app.get("/")
def index():
    return jsonify({"message": "KrishiAI API is running!", "status": "ok"})


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
            "price": float(round(float(r["Price"]), 2)),  # type: ignore
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
        {"district": r["District"], "price": float(round(float(r["Price"]), 2))}  # type: ignore
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
            "price": float(round(float(r["Price"]), 2)),  # type: ignore
        }
        for _, r in out.iterrows()
    ]
    return jsonify({"crop": crop, "items": items, "latest_date": latest_date.date().isoformat()})


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(h))


def _latest_for_crop(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    f = df[df["Crop"].str.lower() == crop.lower()].copy()
    if f.empty:
        return f
    f["Date"] = pd.to_datetime(f["Date"])
    latest_date = f["Date"].max()
    latest = f[f["Date"] == latest_date].copy()
    latest = (
        latest.sort_values(["District", "Date"])
        .groupby(["District"], as_index=False)
        .tail(1)
    )
    return latest


def _detect_crop_and_district(message: str) -> Tuple[str, str]:
    m = (message or "").lower()
    crop = ""
    district = ""

    # crops
    for c in CROPS:
        if c.lower() in m:
            crop = c
            break
    # districts (prefer longer names first)
    for d in sorted(DISTRICTS, key=lambda x: -len(x)):
        if d.lower() in m:
            district = d
            break

    return crop, district


def _chat_help_text() -> str:
    return (
        "I can help with:\n"
        "- Live prices (e.g., 'Tomato price in Mysuru')\n"
        "- AI prediction (e.g., 'predict Onion price in Mandya')\n"
        "- Best market (e.g., 'best market for Potato')\n"
        "- Profit optimizer (e.g., 'profit optimizer for Tomato from Mysuru')\n"
        "- Logistics estimate (e.g., 'logistics for Tomato from Mysuru')\n"
        "- Price alerts (e.g., 'alerts for Tomato')\n"
        "- Demand forecast (e.g., 'demand forecast for Tomato')\n"
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a question. " + _chat_help_text()})

    msg = message.lower()
    crop, district = _detect_crop_and_district(message)
    df = _df_cached()

    # Intent routing
    if "help" in msg or "how" in msg or "use" in msg:
        return jsonify({"reply": _chat_help_text()})

    if "alert" in msg or "increase" in msg:
        if not crop:
            return jsonify({"reply": "Tell me the crop name for alerts (example: 'alerts for Tomato')."})
        res = price_alerts()
        # price_alerts() returns a Flask response; call logic inline instead:
        f = df[df["Crop"].str.lower() == crop.lower()].copy()
        if f.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        f["Date"] = pd.to_datetime(f["Date"])
        f = f.sort_values(["District", "Date"])
        last_two = f.groupby("District").tail(2)
        alerts = []
        for dist, grp in last_two.groupby("District"):
            if len(grp) < 2:
                continue
            prev, curr = grp.iloc[-2], grp.iloc[-1]  # type: ignore
            prev_price = float(prev["Price"])
            curr_price = float(curr["Price"])
            if prev_price <= 0:
                continue
            pct = (curr_price - prev_price) / prev_price * 100.0
            if pct > 10.0:
                alerts.append((pct, dist, curr_price))
        alerts.sort(reverse=True, key=lambda x: x[0])
        if not alerts:
            return jsonify({"reply": f"No sudden price alerts for {crop} today. Prices look stable."})
        top = alerts[:5]  # type: ignore
        lines = [f"Price alerts for {crop} (top districts):"]
        for pct, dist, price in top:
            lines.append(f"- {dist}: Rs {price:.0f} (+{pct:.1f}%)")
        return jsonify({"reply": "\n".join(lines)})

    if "forecast" in msg or "demand forecast" in msg or "demand" in msg:
        if not crop:
            return jsonify({"reply": "Which crop demand should I forecast? (example: 'demand forecast for Tomato')"})
        # reuse demand_forecast logic quickly
        f = df[df["Crop"].str.lower() == crop.lower()].copy()
        if f.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        f["Date"] = pd.to_datetime(f["Date"])
        f = f.sort_values("Date")
        daily = f.groupby("Date", as_index=False)["Price"].mean()
        recent = daily.tail(30)
        if len(recent) < 3:
            return jsonify({"reply": f"Not enough history to forecast demand for {crop}."})
        recent = recent.reset_index(drop=True)
        recent["t"] = range(len(recent))
        x = recent["t"].to_numpy()
        y = recent["Price"].to_numpy()
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        tail5 = recent.tail(5)["Price"].mean()
        prev5 = recent.tail(10).head(5)["Price"].mean()
        delta_pct = (tail5 - prev5) / prev5 * 100.0 if prev5 > 0 else 0.0
        if slope > 0 and delta_pct > 3:
            return jsonify({"reply": f"Demand for {crop} is increasing. Prices rose ~{delta_pct:.1f}% recently."})
        if slope < 0 and delta_pct < -3:
            return jsonify({"reply": f"Demand for {crop} is decreasing. Prices fell ~{abs(delta_pct):.1f}% recently."})
        return jsonify({"reply": f"Demand for {crop} looks stable with small price changes (~{delta_pct:.1f}%)."})

    if "logistics" in msg or "transport" in msg or "distance" in msg:
        if not crop or not district:
            return jsonify(
                {
                    "reply": "Ask like: 'logistics for Tomato from Mysuru' (include crop and your district)."
                }
            )
        if district not in KARNATAKA_COORDS:
            return jsonify({"reply": f"I don't recognize the district '{district}'."})
        latest = _latest_for_crop(df, crop)
        if latest.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        origin = KARNATAKA_COORDS[district]
        rows = []
        for _, r in latest.iterrows():
            dist = r["District"]
            price = float(r["Price"])
            coord = KARNATAKA_COORDS.get(dist)
            km = _haversine_km(origin, coord) if coord else 0.0
            cost = km * 0.05
            net = price - cost
            rows.append((net, dist, price, cost, km))
        rows.sort(reverse=True, key=lambda x: x[0])
        best = rows[0]
        return jsonify(
            {
                "reply": (
                    f"From {district}, best net profit for {crop} is in {best[1]}: "
                    f"price Rs {best[2]:.0f}, transport Rs {best[3]:.2f} (~{best[4]:.0f} km), net Rs {best[0]:.2f}."
                )
            }
        )

    if "profit" in msg or "optimize" in msg:
        if not crop or not district:
            return jsonify(
                {
                    "reply": "Ask like: 'profit optimizer for Tomato from Mysuru' (include crop and your district)."
                }
            )
        if district not in KARNATAKA_COORDS:
            return jsonify({"reply": f"I don't recognize the district '{district}'."})
        latest = _latest_for_crop(df, crop)
        if latest.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        origin = KARNATAKA_COORDS[district]
        profit_rows = []
        for _, r in latest.iterrows():
            dist = r["District"]
            price = float(r["Price"])
            coord = KARNATAKA_COORDS.get(dist)
            km = _haversine_km(origin, coord) if coord else 0.0
            cost = km * 0.05
            net = price - cost
            profit_rows.append((net, dist, price, cost))
        profit_rows.sort(reverse=True, key=lambda x: x[0])
        top = profit_rows[:5]  # type: ignore
        lines = [f"Top profitable markets for {crop} (from {district}):"]
        for net, dist, price, cost in top:
            lines.append(f"- {dist}: price Rs {price:.0f} - transport Rs {cost:.2f} = net Rs {net:.2f}")
        return jsonify({"reply": "\n".join(lines)})

    if "predict" in msg or "prediction" in msg or "future" in msg:
        if not crop or not district:
            return jsonify(
                {
                    "reply": "Ask like: 'predict Tomato price in Mysuru'. Please include crop and district."
                }
            )
        try:
            bundle = _model_for_crop(crop)
            forecast = predict_next(bundle, district=district, horizon_days=1)
            p = forecast[0]["predicted_price"] if forecast else None
            if p is None:
                return jsonify({"reply": "I couldn't generate a prediction right now."})
            return jsonify({"reply": f"Predicted next price for {crop} in {district} is about ₹{p:.0f} per kg."})
        except Exception:
            return jsonify({"reply": f"I couldn't predict for {crop} in {district}. Check crop/district names."})

    if "best market" in msg or "best mandi" in msg or ("best" in msg and "market" in msg):
        if not crop:
            return jsonify({"reply": "Which crop? Example: 'best market for Tomato'."})
        latest = _latest_for_crop(df, crop)
        if latest.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        latest = latest.sort_values("Price", ascending=False)
        best_row = latest.iloc[0]
        return jsonify(
            {
                "reply": f"Best market for {crop} today is {best_row['District']} at around Rs {float(best_row['Price']):.0f} per kg."
            }
        )

    # Default: price question / fallback
    if "price" in msg or crop:
        if not crop:
            return jsonify({"reply": "Tell me the crop and district. Example: 'Tomato price in Mysuru'."})
        latest = _latest_for_crop(df, crop)
        if latest.empty:
            return jsonify({"reply": f"I couldn't find data for {crop}."})
        if district:
            match = latest[latest["District"].str.lower() == district.lower()]
            if match.empty:
                return jsonify({"reply": f"I couldn't find {crop} price for {district}."})
            p = float(match.iloc[0]["Price"])
            return jsonify({"reply": f"Latest {crop} price in {district} is about Rs {p:.0f} per kg."})
        # no district: show top 3
        top = latest.sort_values("Price", ascending=False).head(3)
        lines = [f"Top {crop} prices today:"]
        for _, r in top.iterrows():
            lines.append(f"- {r['District']}: Rs {float(r['Price']):.0f}")
        lines.append("Ask with your district for a specific answer (e.g., 'price in Mysuru').")
        return jsonify({"reply": "\n".join(lines)})

    return jsonify({"reply": "I can help, but I didn’t understand that. " + _chat_help_text()})


@app.get("/profit-optimizer")
def profit_optimizer():
    crop = (request.args.get("crop") or "").strip()
    current_district = (request.args.get("current_district") or "").strip()
    if not crop or not current_district:
        return jsonify({"error": "Missing crop or current_district"}), 400

    df = _df_cached()
    latest = _latest_for_crop(df, crop)
    if latest.empty:
        return jsonify({"error": "Unknown crop"}), 400

    if current_district not in KARNATAKA_COORDS:
        return jsonify({"error": "Unknown current_district"}), 400

    origin = KARNATAKA_COORDS[current_district]
    rows = []
    for _, r in latest.iterrows():
        district = r["District"]
        price = float(round(float(r["Price"]), 2))  # type: ignore
        coord = KARNATAKA_COORDS.get(district)
        if not coord:
            distance_km = 0.0
        else:
            distance_km = _haversine_km(origin, coord)
        transport_cost = distance_km * 0.05
        net_profit = price - transport_cost
        rows.append(
            {
                "district": district,
                "price": price,
                "distance_km": round(distance_km, 1),  # type: ignore
                "transport_cost": round(transport_cost, 2),  # type: ignore
                "net_profit": round(net_profit, 2),  # type: ignore
            }
        )

    rows.sort(key=lambda x: x["net_profit"], reverse=True)
    top = rows[:10]  # type: ignore
    best = top[0] if top else None
    return jsonify(
        {
            "crop": crop,
            "current_district": current_district,
            "recommended": best,
            "markets": top,
        }
    )


@app.get("/price-alerts")
def price_alerts():
    crop = (request.args.get("crop") or "").strip()
    if not crop:
        return jsonify({"error": "Missing crop"}), 400

    df = _df_cached()
    f = df[df["Crop"].str.lower() == crop.lower()].copy()
    if f.empty:
        return jsonify({"alerts": []})

    f["Date"] = pd.to_datetime(f["Date"])
    f = f.sort_values(["District", "Date"])
    # last two entries per district
    last_two = f.groupby("District").tail(2)

    alerts = []
    for district, grp in last_two.groupby("District"):
        if len(grp) < 2:
            continue
        prev, curr = grp.iloc[-2], grp.iloc[-1]  # type: ignore
        prev_price = float(prev["Price"])
        curr_price = float(curr["Price"])
        if prev_price <= 0:
            continue
        pct = (curr_price - prev_price) / prev_price * 100.0
        if pct > 10.0:
            alerts.append(
                {
                    "district": district,
                    "crop": crop,
                    "current_price": round(curr_price, 2),  # type: ignore
                    "percentage_increase": round(pct, 2),  # type: ignore
                    "alert_message": f"Price for {crop} in {district} increased by {pct:.1f}% compared to yesterday.",
                }
            )

    alerts.sort(key=lambda x: x["percentage_increase"], reverse=True)
    return jsonify({"crop": crop, "alerts": alerts})


@app.get("/logistics-estimate")
def logistics_estimate():
    crop = (request.args.get("crop") or "").strip()
    base_district = (request.args.get("district") or "").strip()
    if not crop or not base_district:
        return jsonify({"error": "Missing crop or district"}), 400
    if base_district not in KARNATAKA_COORDS:
        return jsonify({"error": "Unknown district"}), 400

    df = _df_cached()
    latest = _latest_for_crop(df, crop)
    if latest.empty:
        return jsonify({"error": "Unknown crop"}), 400

    origin = KARNATAKA_COORDS[base_district]
    rows = []
    for _, r in latest.iterrows():
        district = r["District"]
        price = float(round(float(r["Price"]), 2))  # type: ignore
        coord = KARNATAKA_COORDS.get(district)
        if not coord:
            distance_km = 0.0
        else:
            distance_km = _haversine_km(origin, coord)
        transport_cost = distance_km * 0.05
        net_profit = price - transport_cost
        rows.append(
            {
                "district": district,
                "price": price,
                "distance_km": round(distance_km, 1),  # type: ignore
                "transport_cost": round(transport_cost, 2),  # type: ignore
                "net_profit": round(net_profit, 2),  # type: ignore
            }
        )

    rows.sort(key=lambda x: x["net_profit"], reverse=True)
    return jsonify(
        {
            "crop": crop,
            "base_district": base_district,
            "estimates": rows,
        }
    )


@app.get("/demand-forecast")
def demand_forecast():
    crop = (request.args.get("crop") or "").strip()
    if not crop:
        return jsonify({"error": "Missing crop"}), 400

    df = _df_cached()
    f = df[df["Crop"].str.lower() == crop.lower()].copy()
    if f.empty:
        return jsonify({"crop": crop, "trend": "unknown", "message": "No data available."})

    f["Date"] = pd.to_datetime(f["Date"])
    f = f.sort_values("Date")
    # Aggregate by date: mean price across districts
    daily = f.groupby("Date", as_index=False)["Price"].mean()
    # focus on recent 30 days
    recent = daily.tail(30)
    if len(recent) < 3:
        return jsonify({"crop": crop, "trend": "unknown", "message": "Not enough history."})

    recent = recent.reset_index(drop=True)
    recent["t"] = range(len(recent))
    x = recent["t"].to_numpy()
    y = recent["Price"].to_numpy()
    # simple linear trend
    slope, intercept = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()), float(
        y.mean()
    )

    # Compare last 5 vs previous 5 days as a sanity check
    tail5 = recent.tail(5)["Price"].mean()
    prev5 = recent.tail(10).head(5)["Price"].mean()
    delta_pct = (tail5 - prev5) / prev5 * 100.0 if prev5 > 0 else 0.0

    if slope > 0 and delta_pct > 3:
        trend = "increasing"
        message = f"Demand for {crop} is trending up: average prices rose by {delta_pct:.1f}% over the last week."
    elif slope < 0 and delta_pct < -3:
        trend = "decreasing"
        message = f"Demand for {crop} seems to be softening: average prices fell by {abs(delta_pct):.1f}% recently."
    else:
        trend = "stable"
        message = f"Demand for {crop} looks relatively stable with minor price movements."

    return jsonify(
        {
            "crop": crop,
            "trend": trend,
            "message": message,
            "recent_window_days": len(recent),
            "change_percent": round(delta_pct, 2),  # type: ignore
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

