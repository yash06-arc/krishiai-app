from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


DISTRICTS = [
    "Bangalore Urban",
    "Bangalore Rural",
    "Mysuru",
    "Mandya",
    "Tumakuru",
    "Kolar",
    "Chikkaballapur",
    "Ramanagara",
    "Hassan",
    "Chikkamagaluru",
    "Kodagu",
    "Dakshina Kannada",
    "Udupi",
    "Uttara Kannada",
    "Shivamogga",
    "Davangere",
    "Chitradurga",
    "Ballari",
    "Koppal",
    "Raichur",
    "Kalaburagi",
    "Yadgir",
    "Bidar",
    "Vijayapura",
    "Bagalkot",
    "Belagavi",
    "Dharwad",
    "Gadag",
    "Haveri",
]

CROPS = [
    "Tomato",
    "Onion",
    "Potato",
    "Chilli",
    "Beans",
    "Cabbage",
    "Cauliflower",
    "Carrot",
    "Brinjal",
    "Capsicum",
    "Cotton",
    "Sugarcane",
    "Maize",
    "Rice",
    "Wheat",
    "Ragi",
    "Groundnut",
    "Sunflower",
    "Turmeric",
    "Ginger",
]


@dataclass(frozen=True)
class DataSpec:
    days: int = 160
    start: date | None = None
    seed: int = 7


def _stable_hash(*parts: str) -> int:
    s = "|".join(parts)
    # python's hash() is salted per process; use deterministic hash
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h *= 16777619
        h &= 0xFFFFFFFF
    return int(h)


def _base_price_for_crop(crop: str) -> float:
    bases = {
        "Tomato": 38,
        "Onion": 32,
        "Potato": 28,
        "Chilli": 120,
        "Beans": 60,
        "Cabbage": 26,
        "Cauliflower": 34,
        "Carrot": 40,
        "Brinjal": 42,
        "Capsicum": 85,
        "Cotton": 70,
        "Sugarcane": 6,
        "Maize": 22,
        "Rice": 28,
        "Wheat": 26,
        "Ragi": 30,
        "Groundnut": 55,
        "Sunflower": 45,
        "Turmeric": 140,
        "Ginger": 165,
    }
    return float(bases.get(crop, 35))


def generate_prices(spec: DataSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    start = spec.start or (date.today() - timedelta(days=spec.days))

    rows = []
    for day_idx in range(spec.days):
        d = start + timedelta(days=day_idx)
        doy = d.timetuple().tm_yday
        seasonal = math.sin(2 * math.pi * doy / 365.25)

        for crop in CROPS:
            base = _base_price_for_crop(crop)
            crop_wave = 1.0 + 0.16 * seasonal
            crop_trend = 1.0 + 0.0012 * day_idx

            for district in DISTRICTS:
                dh = _stable_hash(crop, district) % 1000
                district_bias = (dh / 1000.0 - 0.5) * 0.18  # +/- 9%

                noise = rng.normal(0, 0.06)
                shock = 0.0
                if (dh + day_idx) % 97 == 0:
                    shock = rng.normal(0.10, 0.05)

                price = base * crop_wave * crop_trend * (1.0 + district_bias + noise + shock)
                price = max(2.0, price)

                rows.append(
                    {
                        "Crop": crop,
                        "District": district,
                        "Date": d.isoformat(),
                        "Price": float(round(price, 2)),  # type: ignore
                    }
                )

    return pd.DataFrame(rows)


def ensure_dataset(csv_path: str, spec: DataSpec = DataSpec()) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path
    df = generate_prices(spec)
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(here, "..", "data", "prices_karnataka.csv"))
    ensure_dataset(csv_path)
    print(f"Wrote {csv_path}")

