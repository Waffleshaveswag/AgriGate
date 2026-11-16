"""
build_training_data.py - a train-ready table for LightGBM: one row per US COUNTY x CROP, with REAL labels.

LABELS (USDA NASS, real observations)
  grown_100ac      1 if at least 100 acres harvested (commercial production), NaN if the acreage was withheld
  grown            1 if the 2017 or 2022 Census of Agriculture reports harvested acres of the crop in the county
                   ("(D)" = withheld to protect privacy, still means the crop is grown), else 0       -> classification
  acres            mean harvested acres (2017, 2022) where published                                    -> regression
  area_share       acres / county land area                                                             -> regression
  yield            mean NASS survey yield 2014-2023 (crop's own unit: bu/ac, lb/ac, t/ac, cwt/ac)       -> regression
  yield_rel        yield / national median county yield of that crop (1.0 = typical; comparable across crops)

FEATURES
  soil (county)    USDA NRCS SSURGO, area-weighted surface horizon: pH, organic matter, CEC, water holding, sand,
                   silt, clay, lime, salinity, erodibility (K), slope, share poorly drained
  nutrients        USGS soil geochemistry (OFR 2014-1082), 5 nearest sites to the county centre, distance-weighted:
                   P, K, S, Ca, Mg, Zn, Fe, Mn, Cu, Mo (mg/kg), organic carbon
  climate          NASA POWER climatology 2001-2020: rain, PET, aridity, mean / warmest / coldest month temp,
                   dry months, root-zone soil wetness
  crop traits      plants.csv: pH range, N P K S Ca Mg Zn Fe Mn Cu B Mo need (0-3), N fixing, family, roots,
                   temperature and rain range, perennial
  match features   soil-vs-crop gaps the decision tree uses: pH gap, temperature gap, rain gap, need-minus-supply
                   for every nutrient, salt flag, and the rule tree's own score / verdict

Run:  python3 build_training_data.py   ->  county_crop_train.csv (+ .parquet), feature_columns.json
"""
import json, math, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOIL_DIR = HERE.parent / "soil_dataset"
RAW = HERE / "raw"
# raw/nass_county_labels.tsv = rows of NASS Quick Stats bulk file qs.crops_YYYYMMDD.txt.gz
#   (https://www.nass.usda.gov/datasets/) with AGG_LEVEL_DESC=COUNTY, YEAR 2014-2023, AREA HARVESTED / YIELD
NASS_TSV = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW / "nass_county_labels.tsv"
GAZ_ZIP = RAW / "gaz.zip"                      # Census 2023 Gazetteer counties (centroids, land area)
POWER_GRID = RAW / "power_us_grid.json"        # NASA POWER climatology 0.5° grid (from soil_dataset/build_dataset.py)
sys.path.insert(0, str(SOIL_DIR))
from ssurgo_county import state_query  # noqa: E402  (same folder as this script)

STATES = {"AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13", "ID": "16", "IL": "17",
          "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
          "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
          "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51",
          "WA": "53", "WV": "54", "WI": "55", "WY": "56"}

# NASS census item (AREA HARVESTED, ACRES) -> plant name in plants.csv
CENSUS_CROPS = {
    "CORN, GRAIN - ACRES HARVESTED": "Corn", "SOYBEANS - ACRES HARVESTED": "Soybeans", "WHEAT - ACRES HARVESTED": "Wheat",
    "HAY - ACRES HARVESTED": "Hay", "COTTON - ACRES HARVESTED": "Cotton", "RICE - ACRES HARVESTED": "Rice",
    "POTATOES - ACRES HARVESTED": "Potatoes", "SUGARBEETS - ACRES HARVESTED": "Sugar Beets", "PEANUTS - ACRES HARVESTED": "Peanuts",
    "TOMATOES, IN THE OPEN - ACRES HARVESTED": "Tomatoes", "SUGARCANE, SUGAR - ACRES HARVESTED": "Sugarcane",
    "LETTUCE - ACRES HARVESTED": "Lettuce", "CABBAGE, HEAD - ACRES HARVESTED": "Cabbage", "BROCCOLI - ACRES HARVESTED": "Broccoli",
    "CAULIFLOWER - ACRES HARVESTED": "Cauliflowers", "SPINACH - ACRES HARVESTED": "Spinach", "CUCUMBERS - ACRES HARVESTED": "Cucumbers",
    "MELONS, WATERMELON - ACRES HARVESTED": "Watermelon", "EGGPLANT - ACRES HARVESTED": "Eggplants",
    "PEPPERS, CHILE - ACRES HARVESTED": "Chilli Peppers", "PEAS, GREEN, (EXCL SOUTHERN) - ACRES HARVESTED": "Green Peas",
    "ASPARAGUS - ACRES HARVESTED": "Asparagus", "BEETS - ACRES HARVESTED": "Beet", "GREENS, KALE - ACRES HARVESTED": "Kale",
    "ESCAROLE & ENDIVE - ACRES HARVESTED": "Endive", "WATERCRESS - ACRES HARVESTED": "Cress",
    # more census crops added later
    "OATS - ACRES HARVESTED": "Oats", "BARLEY - ACRES HARVESTED": "Barley", "RYE - ACRES HARVESTED": "Rye",
    "TRITICALE - ACRES HARVESTED": "Triticale", "SORGHUM, GRAIN - ACRES HARVESTED": "Sorghum", "MILLET, PROSO - ACRES HARVESTED": "Millet",
    "SWEET CORN - ACRES HARVESTED": "Sweet Corn", "SUNFLOWER - ACRES HARVESTED": "Sunflowers", "SAFFLOWER - ACRES HARVESTED": "Safflower",
    "CANOLA - ACRES HARVESTED": "Canola", "MUSTARD, SEED - ACRES HARVESTED": "Mustard", "FLAXSEED - ACRES HARVESTED": "Flax",
    "BUCKWHEAT - ACRES HARVESTED": "Buckwheat", "BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA) - ACRES HARVESTED": "Dry Beans",
    "PEAS, DRY EDIBLE - ACRES HARVESTED": "Dry Peas", "LENTILS - ACRES HARVESTED": "Lentils", "CHICKPEAS - ACRES HARVESTED": "Chickpeas",
    "BEANS, SNAP - ACRES HARVESTED": "Snap Beans", "SQUASH - ACRES HARVESTED": "Squash", "PUMPKINS - ACRES HARVESTED": "Pumpkins",
    "MELONS, CANTALOUP - ACRES HARVESTED": "Cantaloupe", "PEPPERS, BELL - ACRES HARVESTED": "Bell Peppers",
    "ONIONS, DRY - ACRES HARVESTED": "Onions", "GARLIC - ACRES HARVESTED": "Garlic", "CARROTS - ACRES HARVESTED": "Carrots",
    "CELERY - ACRES HARVESTED": "Celery", "RADISHES - ACRES HARVESTED": "Radishes", "TURNIPS - ACRES HARVESTED": "Turnips",
    "BRUSSELS SPROUTS - ACRES HARVESTED": "Brussels Sprouts", "OKRA - ACRES HARVESTED": "Okra", "SWEET POTATOES - ACRES HARVESTED": "Sweet Potatoes",
}
# NASS survey yield item -> plant (each crop keeps its own unit; yield_rel makes them comparable)
YIELD_ITEMS = {
    "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE": "Corn", "SOYBEANS - YIELD, MEASURED IN BU / ACRE": "Soybeans",
    "WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE": "Wheat", "HAY, (EXCL ALFALFA) - YIELD, MEASURED IN TONS / ACRE": "Hay",
    "COTTON, UPLAND - YIELD, MEASURED IN LB / ACRE": "Cotton", "RICE - YIELD, MEASURED IN LB / ACRE": "Rice",
    "SUGARBEETS - YIELD, MEASURED IN TONS / ACRE": "Sugar Beets", "PEANUTS - YIELD, MEASURED IN LB / ACRE": "Peanuts",
    "POTATOES - YIELD, MEASURED IN CWT / ACRE": "Potatoes", "TOMATOES, IN THE OPEN, PROCESSING - YIELD, MEASURED IN TONS / ACRE": "Tomatoes",
    "SUGARCANE, SUGAR - YIELD, MEASURED IN TONS / ACRE": "Sugarcane",
}
YIELD_UNIT = {v: re.search(r"MEASURED IN (.*)", k).group(1).lower() for k, v in YIELD_ITEMS.items()}
PERENNIAL = {"Grapes", "Asparagus", "Coffee", "Sugarcane", "Hay", "Strawberry"}
SALT_SENSITIVE = {"Strawberry", "Dry Beans", "Snap Beans", "Onions", "Carrots", "Celery", "Radishes", "Bell Peppers", "Cantaloupe", "Squash", "Pumpkins", "Sweet Corn", "Sweet Potatoes", "Okra", "Garlic", "Lentils", "Chickpeas", "Dry Peas", "Flax", "Buckwheat", "Green Peas", "Lettuce", "Radicchio", "Endive", "Cress", "Cucumbers", "Grapes", "Potatoes", "Corn",
                  "Chilli Peppers", "Eggplants", "Cabbage", "Broccoli", "Cauliflowers", "Rice", "Peanuts", "Watermelon"}
NEED = {"low": 1, "medium": 2, "high": 3, "fixes own N": 0}
NUTS = ["N", "P", "K", "S", "Ca", "Mg", "Zn", "Fe", "Mn", "Cu", "Mo"]
SOIL_COL = {"P": "P_mgkg", "K": "K_mgkg", "S": "S_mgkg", "Ca": "Ca_mgkg", "Mg": "Mg_mgkg", "Zn": "Zn_mgkg", "Fe": "Fe_mgkg",
            "Mn": "Mn_mgkg", "Cu": "Cu_mgkg", "Mo": "Mo_mgkg"}


def log(*a):
    print(*a, flush=True)


def num(v):
    v = str(v).replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan                                  # "(D)" withheld, "(Z)" < half unit, "(NA)"


def main():
    # ---------------------------------------------------------------- counties
    with zipfile.ZipFile(GAZ_ZIP) as z:
        g = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g = g[g.USPS.isin(STATES)].rename(columns={"GEOID": "fips", "NAME": "county", "USPS": "state", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    g["land_acres"] = g.ALAND * 0.000247105
    counties = g[["fips", "state", "county", "lat", "lon", "land_acres"]].reset_index(drop=True)
    log(f"{len(counties)} counties in the 48 states")

    # ---------------------------------------------------------------- SSURGO soil per county
    cache = HERE / "ssurgo_county_cache.csv"
    if cache.exists():
        ss = pd.read_csv(cache, dtype={"fips": str})
    else:
        rows = []
        for st, sf in STATES.items():
            for r in state_query(st):
                code = r["areasymbol"][2:]
                if code.isdigit():
                    r["fips"] = sf + code.zfill(3)
                    rows.append(r)
            log(f"  SSURGO {st}: {sum(1 for r in rows if r['fips'].startswith(sf))} survey areas")
        ss = pd.DataFrame(rows)
        for c in ss.columns:
            if c not in ("areasymbol", "fips"):
                ss[c] = pd.to_numeric(ss[c], errors="coerce")
        ss.to_csv(cache, index=False)
    ss = ss.drop(columns=["areasymbol", "acres"], errors="ignore").add_prefix("soil_").rename(columns={"soil_fips": "fips"})
    ss["soil_awc_mm_per_m"] = ss.pop("soil_awc") * 1000
    df = counties.merge(ss, on="fips", how="left")
    df["soil_source_ssurgo"] = df.soil_ph.notna().astype(int)

    # ---------------------------------------------------------------- USGS nutrients + fallback soil, nearest sites
    sites = pd.read_csv(SOIL_DIR / "soil_sites.csv", low_memory=False)
    sites = sites.dropna(subset=["P_mgkg", "K_mgkg"])
    lat0, lon0 = np.radians(sites.lat.values), np.radians(sites.lon.values)
    feats = []
    for la, lo in zip(df.lat, df.lon):
        a = np.sin((lat0 - math.radians(la)) / 2) ** 2 + np.cos(lat0) * math.cos(math.radians(la)) * np.sin((lon0 - math.radians(lo)) / 2) ** 2
        d = 12742 * np.arcsin(np.sqrt(a))
        idx = np.argsort(d)[:5]
        w = 1 / (d[idx] + 1) ** 2
        row = {"nearest_site_km": float(d[idx[0]])}
        for col in list(SOIL_COL.values()) + ["org_carbon_pct", "ph", "sand_pct", "clay_pct", "cec_cmol_kg"]:
            v = sites[col].values[idx].astype(float)
            ok = ~np.isnan(v)
            row["usgs_" + col] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        feats.append(row)
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)
    for a, b in (("soil_ph", "usgs_ph"), ("soil_sand_pct", "usgs_sand_pct"), ("soil_clay_pct", "usgs_clay_pct"), ("soil_cec", "usgs_cec_cmol_kg")):
        df[a] = df[a].fillna(df[b])                    # counties without SSURGO aggregates: FAO HWSD values at nearby sites
    df = df.drop(columns=["usgs_ph", "usgs_sand_pct", "usgs_clay_pct", "usgs_cec_cmol_kg"])
    df["soil_om_pct"] = df.soil_om_pct.fillna(df.usgs_org_carbon_pct * 1.72)
    df["soil_N_est_mgkg"] = df.soil_om_pct * 500       # total N ≈ organic matter / 20

    # ---------------------------------------------------------------- NASA POWER climate
    grid = json.loads(POWER_GRID.read_text())
    keys = np.array([[float(x) for x in k.split(",")] for k in grid])
    DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    sys.path.insert(0, str(SOIL_DIR))
    from build_dataset import hargreaves_pet
    clim = []
    for la, lo in zip(df.lat, df.lon):
        j = int(np.argmin((keys[:, 0] - la) ** 2 + (keys[:, 1] - lo) ** 2))
        gg = grid[f"{keys[j, 0]:.3f},{keys[j, 1]:.3f}"]
        rain = [v * d for v, d in zip(gg["PRECTOTCORR"], DAYS)]
        pet = hargreaves_pet(la, gg["T2M"], gg["T2M_RANGE"])
        clim.append(dict(clim_rain_mm=sum(rain), clim_pet_mm=sum(pet), clim_aridity=sum(rain) / sum(pet), clim_t_mean=np.mean(gg["T2M"]),
                         clim_t_warmest=max(gg["T2M"]), clim_t_coldest=min(gg["T2M"]), clim_dry_months=sum(r < 0.5 * p for r, p in zip(rain, pet)),
                         clim_months_above_10c=sum(t >= 10 for t in gg["T2M"]), clim_root_wetness=float(np.mean(gg["GWETROOT"])),
                         clim_summer_rain_mm=sum(rain[4:9])))
    df = pd.concat([df, pd.DataFrame(clim)], axis=1)
    log("soil, nutrients and climate joined")

    # ---------------------------------------------------------------- labels from NASS
    nass = pd.read_csv(NASS_TSV, sep="\t", dtype=str, usecols=["SOURCE_DESC", "SHORT_DESC", "STATE_ALPHA", "STATE_FIPS_CODE", "COUNTY_ANSI", "YEAR", "VALUE"])
    nass = nass[nass.STATE_ALPHA.isin(STATES) & nass.COUNTY_ANSI.notna()]
    nass["fips"] = nass.STATE_FIPS_CODE.str.zfill(2) + nass.COUNTY_ANSI.str.zfill(3)
    nass["v"] = nass.VALUE.map(num)
    cen = nass[(nass.SOURCE_DESC == "CENSUS") & nass.SHORT_DESC.isin(CENSUS_CROPS)].copy()
    cen["crop"] = cen.SHORT_DESC.map(CENSUS_CROPS)
    area = cen.groupby(["fips", "crop"]).agg(acres=("v", "mean"), census_reports=("v", "size")).reset_index()
    sur = nass[(nass.SOURCE_DESC == "SURVEY") & nass.SHORT_DESC.isin(YIELD_ITEMS)].copy()
    sur["crop"] = sur.SHORT_DESC.map(YIELD_ITEMS)
    yld = sur.dropna(subset=["v"]).groupby(["fips", "crop"]).agg(yield_=("v", "mean"), yield_years=("v", "size")).reset_index()
    yld["yield_rel"] = yld.yield_ / yld.groupby("crop").yield_.transform("median")
    log(f"labels: {len(area)} county-crop census records, {len(yld)} county-crop yield records")

    # ---------------------------------------------------------------- county x crop grid + crop traits
    plants = pd.read_csv(SOIL_DIR / "plants.csv")
    plants = plants[plants.name.isin(set(CENSUS_CROPS.values()))].copy()
    crops = plants.rename(columns={"name": "crop"})
    trait_cols = ["crop", "ph_min", "ph_max", "N_kg_ha", "P_kg_ha", "K_kg_ha", "temp_min_c", "temp_max_c", "rain_min_mm", "rain_max_mm",
                  "family", "fixes_n", "roots", "height"] + [f"{n}_hunger" for n in NUTS + ["B"]]
    crops = crops[trait_cols].copy()
    for n in NUTS + ["B"]:
        crops[f"need_{n}"] = crops.pop(f"{n}_hunger").map(NEED)
    crops["perennial"] = crops.crop.isin(PERENNIAL).astype(int)
    crops["salt_sensitive"] = crops.crop.isin(SALT_SENSITIVE).astype(int)
    crops = crops.add_prefix("crop_").rename(columns={"crop_crop": "crop"})
    t = df.merge(crops, how="cross")
    t = t.merge(area, on=["fips", "crop"], how="left").merge(yld, on=["fips", "crop"], how="left")
    t["grown"] = t.census_reports.notna().astype(int)
    # commercial presence: >= 100 harvested acres (NaN when the census withheld the acreage)
    t["grown_100ac"] = np.where(t.grown == 0, 0, np.where(t.acres >= 100, 1, np.where(t.acres.notna(), 0, np.nan)))
    t["area_share"] = t.acres / t.land_acres
    t = t.rename(columns={"yield_": "yield"})
    t["yield_unit"] = t.crop.map(YIELD_UNIT)

    # ---------------------------------------------------------------- engineered soil-vs-crop match features
    t["m_ph_gap"] = np.maximum(0, np.maximum(t.crop_ph_min - t.soil_ph, t.soil_ph - t.crop_ph_max))
    t["m_ph_mid_diff"] = t.soil_ph - (t.crop_ph_min + t.crop_ph_max) / 2
    t["m_temp_gap"] = np.maximum(0, np.maximum(t.crop_temp_min_c - t.clim_t_warmest, t.clim_t_warmest - t.crop_temp_max_c))
    t["m_rain_ratio"] = t.clim_rain_mm / t.crop_rain_min_mm
    t["m_salt_risk"] = ((t.soil_ec_dsm > 4) & (t.crop_salt_sensitive == 1)).astype(int)
    soil_amount = {"N": "soil_N_est_mgkg", **{k: "usgs_" + v for k, v in SOIL_COL.items()}}
    for n, col in soil_amount.items():
        q1, q2 = df[col].quantile([1 / 3, 2 / 3])
        lvl = np.where(t[col] < q1, 1, np.where(t[col] > q2, 3, 2))
        if n in ("Fe", "Mn", "Zn", "Cu", "P"):
            lvl = np.where(t.soil_ph > 7.5, np.maximum(1, lvl - 1), lvl)       # high pH locks these up
        if n in ("Mo", "Ca", "Mg", "P"):
            lvl = np.where(t.soil_ph < 5.5, np.maximum(1, lvl - 1), lvl)       # low pH locks these up
        t[f"soil_level_{n}"] = lvl
        t[f"m_need_minus_supply_{n}"] = t[f"crop_need_{n}"] - lvl
    t["m_nutrient_shortfalls"] = (t[[f"m_need_minus_supply_{n}" for n in NUTS]] >= 2).sum(axis=1)
    # the hand-written decision tree, as features (LightGBM can learn where the rules are right or wrong)
    t["rule_soil_ok"] = ((t.m_ph_gap <= 1.0) & (t.m_salt_risk == 0)).astype(int)
    t["rule_score"] = (40 * np.clip(1 - t.m_ph_gap, 0, 1)
                       + 60 * (1 - np.clip(t[[f"m_need_minus_supply_{n}" for n in NUTS]], 0, None).mean(axis=1) / 2)).round(2)

    # ---------------------------------------------------------------- write
    cat = ["state", "crop", "crop_family", "crop_roots", "crop_height"]
    label = ["grown", "grown_100ac", "acres", "area_share", "yield", "yield_rel", "yield_unit", "yield_years", "census_reports"]
    ids = ["fips", "county", "lat", "lon", "land_acres"]
    features = [c for c in t.columns if c not in label + ids and c != "state"]
    t = t[ids + ["state"] + features + label]
    for c in cat:
        t[c] = t[c].astype("category")
    t = t.copy()
    t.to_csv(HERE / "county_crop_train.csv", index=False)
    try:
        t.to_parquet(HERE / "county_crop_train.parquet", index=False)
    except Exception as e:
        log("parquet skipped:", e)
    json.dump({"features": features, "categorical": [c for c in cat if c in features], "labels": label, "ids": ids,
               "group_for_cv": "state"}, open(HERE / "feature_columns.json", "w"), indent=1)
    log(f"wrote county_crop_train.csv: {len(t):,} rows ({t.fips.nunique()} counties x {t.crop.nunique()} crops), "
        f"{len(features)} features; grown=1 for {t.grown.mean():.1%}; yield rows {t['yield'].notna().sum():,}")


if __name__ == "__main__":
    main()
