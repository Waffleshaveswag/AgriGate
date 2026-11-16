"""
agrigate_model.py - the trained LightGBM models behind AgriGate's crop recommendations.

  predict(lat, lon)          -> every crop's chance of being grown commercially in the county at that point,
                                plus predicted yield vs the national typical (and the real USDA values)
  future(lat, lon, year)     -> NASA POWER 1984-2025 linear trend at the point -> projected climate -> model re-run

Models: model/models/lightgbm_grown_tuned.txt, lightgbm_yield_tuned.txt (LightGBM, tuned with 5-fold CV grouped by state)
Training data: NASA POWER climate + USDA SSURGO soil + USGS soil nutrients, with USDA NASS county crop data as the answers.

The saved LightGBM trees are run by tree_model.py (plain Python), so the LightGBM library is not needed to use the
website: only to retrain. The predictions are the same as LightGBM's.
"""
import json, math, ssl, urllib.parse, urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tree_model import TreeModel

HERE = Path(__file__).resolve().parent
try:                                   # trusted certificates for https, so NASA POWER works on any computer
    import certifi                     # (Python from python.org on a Mac has none until an extra install step)
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = None
META = json.load(open(HERE / "data" / "feature_columns.json"))
FEAT, CAT = META["features"], META["categorical"]
RESULTS = json.load(open(HERE / "data" / "final_results.json"))
DATA = pd.read_parquet(HERE / "data" / "county_crop_train.parquet")
GROWN = TreeModel(HERE / "models" / "lightgbm_grown_tuned.txt")
YIELD = TreeModel(HERE / "models" / "lightgbm_yield_tuned.txt")
CAT_COLS = [f for f in FEAT if f in CAT]             # categorical features, in the order the model saw them


def _run(model, frame):
    """Run a model on rows of the training table (text categories become the codes the model was trained with)."""
    X = frame[FEAT].copy()
    for col, names in zip(CAT_COLS, model.pandas_categorical):
        code = {name: i for i, name in enumerate(names)}
        X[col] = X[col].astype(str).map(code).astype(float)
    return model.predict(X.astype(float).to_numpy().tolist())


# The 17 model crops that also exist on the website (the other 9 census crops have no match in the site's crop list)
SITE_CROPS = {"Broccoli", "Chilli Peppers", "Corn", "Cotton", "Cress", "Endive", "Hay", "Kale", "Lettuce", "Peanuts",
              "Potatoes", "Rice", "Soybeans", "Spinach", "Sugar Beets", "Tomatoes", "Wheat"}
YIELD_CROPS = set(DATA.loc[DATA.yield_rel.notna(), "crop"].astype(str))
COUNTIES = DATA.drop_duplicates("fips")[["fips", "state", "county", "lat", "lon"]].reset_index(drop=True)
MODEL_INFO = {"name": "LightGBM (tuned)",
              "median_per_crop_auc": round(RESULTS["grown_auc_median_per_crop"]["lightgbm_tuned"], 2),
              "rule_baseline_auc": round(RESULTS["grown_auc_median_per_crop"]["rule_tree"], 2),
              "yield_r2": round(RESULTS["yield_r2"]["lightgbm_tuned"], 2),
              "tested_on": "states the model never saw (5-fold, grouped by state)",
              "trained_on": "NASA POWER climate + USDA SSURGO soil + USGS nutrients; labels from USDA NASS (3,108 counties x 26 crops)"}


def _num(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else round(v, 4)


def in_us(lat, lon):
    return 24 <= lat <= 50 and -125 <= lon <= -66


def county_rows(lat, lon):
    d = (COUNTIES.lat - lat) ** 2 + ((COUNTIES.lon - lon) * math.cos(math.radians(lat))) ** 2
    c = COUNTIES.loc[d.idxmin()]
    rows = DATA[(DATA.fips == c.fips) & DATA.crop.astype(str).isin(SITE_CROPS)].copy()
    rows["chance_grown"] = _run(GROWN, rows)
    rows["yield_vs_typical"] = _run(YIELD, rows)
    county = {"fips": c.fips, "name": c.county, "state": c.state, "distance_km": round(math.sqrt(d.min()) * 111, 1)}
    return county, rows


def predict(lat, lon):
    if not in_us(lat, lon):
        return {"ok": False, "error": "The LightGBM model covers the 48 contiguous US states only."}
    county, rows = county_rows(lat, lon)
    rows = rows.sort_values("chance_grown", ascending=False)
    return {"ok": True, "county": county, "model": MODEL_INFO,
            "crops": [{"crop": str(r.crop), "chance_grown": _num(r.chance_grown),
                       "yield_vs_typical": _num(r.yield_vs_typical) if str(r.crop) in YIELD_CROPS else None,
                       "usda_grown_100ac": _num(r.grown_100ac), "usda_yield_vs_typical": _num(r.yield_rel)}
                      for r in rows.itertuples()]}


@lru_cache(maxsize=256)
def _power_history(lat, lon):
    q = dict(parameters="T2M,PRECTOTCORR", community="AG", latitude=lat, longitude=lon, start=1984, end=2025, format="JSON")
    with urllib.request.urlopen("https://power.larc.nasa.gov/api/temporal/monthly/point?" + urllib.parse.urlencode(q), timeout=60, context=_SSL) as r:
        p = json.load(r)["properties"]["parameter"]
    days = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ys, ts, rs = [], [], []
    for y in range(1984, 2026):
        t = [p["T2M"].get(f"{y}{m:02d}") for m in range(1, 13)]
        pr = [p["PRECTOTCORR"].get(f"{y}{m:02d}") for m in range(1, 13)]
        if None in t or None in pr or min(t) <= -999:
            continue
        ys.append(y); ts.append(sum(t) / 12); rs.append(sum(v * d for v, d in zip(pr, days)))
    return np.array(ys), np.array(ts), np.array(rs)


def future(lat, lon, year=2050):
    if not in_us(lat, lon):
        return {"ok": False, "error": "The LightGBM model covers the 48 contiguous US states only."}
    try:
        yrs, temp, rain = _power_history(round(lat, 2), round(lon, 2))
    except Exception as e:
        return {"ok": False, "error": f"NASA POWER unreachable ({e.__class__.__name__})"}
    ta, tb = np.polyfit(yrs, temp, 1)
    ra, rb = np.polyfit(yrs, rain, 1)
    dT = float(ta * (year - 2010))
    dR = float((ra * year + rb) / (ra * 2010 + rb) - 1)
    county, now = county_rows(lat, lon)
    fut = now.copy()
    for col in ("clim_t_mean", "clim_t_warmest", "clim_t_coldest"):
        fut[col] += dT
    for col in ("clim_rain_mm", "clim_summer_rain_mm"):
        fut[col] *= (1 + dR)
    fut["clim_pet_mm"] = now.clim_pet_mm * (fut.clim_t_mean + 17.8) / (now.clim_t_mean + 17.8)
    fut["clim_aridity"] = fut.clim_rain_mm / fut.clim_pet_mm
    fut["clim_months_above_10c"] = np.minimum(12, fut.clim_months_above_10c + np.round(dT / 3))
    fut["m_temp_gap"] = np.maximum(0, np.maximum(fut.crop_temp_min_c - fut.clim_t_warmest, fut.clim_t_warmest - fut.crop_temp_max_c))
    fut["m_rain_ratio"] = fut.clim_rain_mm / fut.crop_rain_min_mm
    after = _run(GROWN, fut)
    out = pd.DataFrame({"crop": now.crop.astype(str).values, "now": now.chance_grown.values, "future": after})
    out["change"] = out.future - out.now
    return {"ok": True, "county": county,
            "nasa_power_trend": {"years": f"{yrs.min()}-{yrs.max()}", "warming_c_per_decade": round(float(ta * 10), 3),
                                 "rain_mm_per_decade": round(float(ra * 10), 1)},
            "projection": {"year": year, "warming_c_vs_2001_2020": round(dT, 2), "rain_change_pct": round(dR * 100, 1)},
            "crops": [{"crop": r.crop, "chance_now": _num(r.now), "chance_future": _num(r.future), "change": _num(r.change)}
                      for r in out.sort_values("change").itertuples()]}


def climate_trend(lat, lon, until=2050):
    """NASA POWER yearly mean temperature + rain 1984-2025 at the point, with least-squares linear regression lines
    and a projection to `until`. Works anywhere on Earth (NASA POWER is global)."""
    try:
        yrs, temp, rain = _power_history(round(lat, 2), round(lon, 2))
    except Exception as e:
        return {"ok": False, "error": f"NASA POWER unreachable ({e.__class__.__name__})"}
    out = {"ok": True, "source": f"NASA POWER monthly, {int(yrs.min())}-{int(yrs.max())}", "years": yrs.tolist()}
    for name, y, unit in (("temperature", temp, "°C"), ("rain", rain, "mm")):
        a, b = np.polyfit(yrs, y, 1)
        fit = a * yrs + b
        resid = y - fit
        r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        sd = float(np.sqrt((resid ** 2).sum() / (len(y) - 2)))
        out[name] = {"unit": unit, "values": [round(float(v), 2) for v in y], "slope_per_decade": round(float(a * 10), 3),
                     "intercept": float(b), "slope": float(a), "r2": round(float(r2), 3), "resid_sd": round(sd, 3),
                     "now": round(float(a * yrs.max() + b), 2), "projected": round(float(a * until + b), 2), "projected_year": until}
    return out
