"""
build_plants.py - one plant requirements table for the decision tree.

  Kaggle "Soil Nutrients" (snmahsa/soil-nutrients): 22 vegetables/fruits x 700 rows each. The rows are SIMULATED around
      one recipe per crop, so we summarize each crop's recipe: pH range, N / P / K need, temperature, rain, humidity,
      light, soil type, season, photoperiod. (Not used to train a yield model: yield is random noise within each crop.)
  Team research doc: 12 field crops (+31 more USDA census crops from extension guides / FAO EcoCrop) with ideal pH and N, P2O5, K2O, S in kg/ha (Oregon State, Texas A&M, LSU, Montana State).
  MSU Extension E-486: each plant's need for S, Ca, Mg, Zn, Fe, Mn, Cu, B, Mo (low / medium / high).
  Added botany: plant family (shared pests/diseases), nitrogen fixing, root depth, height, needed for the compatibility tree.

Run:  python3 build_plants.py "path/to/Soil Nutrients.csv"   -> plants.csv
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "raw" / "Soil Nutrients.csv"

# family, fixes N, roots, height  (common extension-guide values)
BOTANY = {
    "Strawberry": ("rose", 0, "shallow", "short"), "Watermelon": ("cucurbit", 0, "deep", "short"), "Grapes": ("grape", 0, "deep", "tall"),
    "Arugula": ("brassica", 0, "shallow", "short"), "Beet": ("amaranth", 0, "medium", "short"), "Chard": ("amaranth", 0, "medium", "short"),
    "Cress": ("brassica", 0, "shallow", "short"), "Endive": ("aster", 0, "medium", "short"), "Kale": ("brassica", 0, "medium", "medium"),
    "Lettuce": ("aster", 0, "shallow", "short"), "Radicchio": ("aster", 0, "medium", "short"), "Spinach": ("amaranth", 0, "shallow", "short"),
    "Tomatoes": ("nightshade", 0, "deep", "medium"), "Eggplants": ("nightshade", 0, "medium", "medium"), "Asparagus": ("asparagus", 0, "deep", "tall"),
    "Chilli Peppers": ("nightshade", 0, "medium", "medium"), "Cabbage": ("brassica", 0, "medium", "short"), "Cucumbers": ("cucurbit", 0, "medium", "short"),
    "Potatoes": ("nightshade", 0, "shallow", "short"), "Cauliflowers": ("brassica", 0, "medium", "short"), "Broccoli": ("brassica", 0, "medium", "medium"),
    "Green Peas": ("legume", 1, "medium", "medium"),
}

# Team research doc (kg/ha per season, ranges -> middle value) + typical temperature / rain needs (FAO crop sheets)
FIELD = [
    # name,        pH,          N,          P2O5,     S,        K2O,       family,   fixN, roots,   height,  temp(°C),  rain(mm),    seasons
    ("Corn",       (6.0, 6.5), (120, 200), (40, 80), (15, 25), (60, 120), "grass",   0, "deep",   "tall",   (18, 32), (500, 800),  "Spring,Summer"),
    ("Soybeans",   (6.0, 6.8), (0, 30),    (40, 60), (10, 20), (60, 100), "legume",  1, "medium", "medium", (20, 30), (450, 700),  "Spring,Summer"),
    ("Hay",        (6.0, 7.0), (50, 150),  (30, 60), (10, 25), (150, 300), "legume", 1, "deep",   "medium", (15, 28), (800, 1600), "Spring,Summer,Fall"),
    ("Wheat",      (6.0, 7.0), (80, 150),  (30, 60), (10, 20), (30, 60),  "grass",   0, "deep",   "medium", (12, 24), (450, 650),  "Fall,Winter,Spring"),
    ("Cotton",     (5.8, 6.5), (80, 150),  (30, 60), (15, 30), (60, 120), "mallow",  0, "deep",   "medium", (21, 32), (700, 1300), "Spring,Summer"),
    ("Rice",       (5.5, 6.5), (80, 150),  (30, 60), (10, 20), (40, 80),  "grass",   0, "shallow", "medium", (20, 35), (900, 2000), "Summer"),
    ("Sugar Beets", (6.0, 7.0), (80, 140), (30, 60), (15, 30), (100, 200), "amaranth", 0, "deep",  "short",  (15, 24), (550, 750),  "Spring,Summer"),
    ("Peanuts",    (5.8, 6.5), (0, 20),    (30, 60), (15, 30), (30, 60),  "legume",  1, "medium", "short",  (22, 30), (500, 700),  "Spring,Summer"),
    ("Coffee",     (5.0, 6.0), (100, 200), (20, 50), (15, 30), (100, 200), "coffee", 0, "deep",   "tall",   (18, 24), (1200, 1800), "All year"),
    ("Sugarcane",  (6.0, 7.0), (100, 200), (30, 80), (15, 30), (100, 200), "grass",  0, "deep",   "tall",   (22, 32), (1500, 2500), "All year"),
    # added later: more USDA census crops (typical values from state extension guides and FAO EcoCrop)
    ("Oats",       (5.5, 7.0), (50, 90),   (30, 50), (10, 20), (30, 60),  "grass",   0, "medium", "medium", (10, 24), (400, 800),  "Spring,Fall"),
    ("Barley",     (6.0, 7.5), (60, 110),  (30, 50), (10, 20), (30, 60),  "grass",   0, "deep",   "medium", (10, 24), (350, 650),  "Spring,Fall"),
    ("Rye",        (5.0, 7.0), (40, 90),   (20, 40), (10, 20), (20, 50),  "grass",   0, "deep",   "tall",   (8, 22),  (400, 800),  "Fall,Winter,Spring"),
    ("Triticale",  (5.5, 7.5), (60, 110),  (30, 50), (10, 20), (30, 60),  "grass",   0, "deep",   "medium", (10, 24), (400, 750),  "Fall,Winter,Spring"),
    ("Sorghum",    (5.5, 7.5), (60, 130),  (30, 60), (10, 20), (30, 60),  "grass",   0, "deep",   "tall",   (22, 34), (400, 700),  "Spring,Summer"),
    ("Millet",     (5.5, 7.5), (40, 70),   (20, 40), (10, 15), (20, 40),  "grass",   0, "shallow", "medium", (20, 30), (250, 500), "Summer"),
    ("Sweet Corn", (6.0, 6.8), (120, 180), (50, 90), (15, 25), (80, 140), "grass",   0, "deep",   "tall",   (18, 30), (500, 800),  "Spring,Summer"),
    ("Sunflowers", (6.0, 7.5), (60, 110),  (30, 50), (10, 25), (40, 80),  "aster",   0, "deep",   "tall",   (18, 30), (400, 700),  "Spring,Summer"),
    ("Safflower",  (6.0, 8.0), (50, 100),  (20, 40), (10, 20), (20, 40),  "aster",   0, "deep",   "medium", (20, 32), (300, 500),  "Spring,Summer"),
    ("Canola",     (5.8, 7.5), (90, 150),  (30, 50), (20, 35), (30, 60),  "brassica", 0, "deep",  "medium", (12, 24), (400, 650),  "Spring,Fall"),
    ("Mustard",    (6.0, 7.5), (60, 100),  (20, 40), (15, 30), (20, 40),  "brassica", 0, "medium", "medium", (12, 24), (300, 500), "Spring,Summer"),
    ("Flax",       (6.0, 7.0), (50, 90),   (20, 40), (10, 20), (20, 40),  "flax",    0, "shallow", "medium", (10, 22), (350, 600), "Spring,Summer"),
    ("Buckwheat",  (5.0, 7.0), (30, 60),   (20, 40), (5, 15),  (20, 40),  "knotweed", 0, "shallow", "medium", (15, 25), (400, 700), "Summer"),
    ("Dry Beans",  (6.0, 7.0), (20, 60),   (30, 50), (10, 20), (40, 80),  "legume",  1, "medium", "short",  (18, 28), (350, 600),  "Spring,Summer"),
    ("Dry Peas",   (6.0, 7.5), (0, 20),    (20, 40), (10, 20), (20, 50),  "legume",  1, "medium", "medium", (10, 22), (350, 600),  "Spring"),
    ("Lentils",    (6.0, 7.5), (0, 20),    (20, 40), (10, 20), (20, 40),  "legume",  1, "medium", "short",  (12, 24), (300, 500),  "Spring"),
    ("Chickpeas",  (6.0, 8.0), (0, 20),    (20, 40), (10, 20), (20, 40),  "legume",  1, "deep",   "short",  (15, 28), (300, 500),  "Spring"),
    ("Snap Beans", (6.0, 7.0), (30, 70),   (40, 80), (10, 20), (60, 100), "legume",  1, "medium", "short",  (18, 28), (400, 700),  "Spring,Summer"),
    ("Squash",     (6.0, 7.0), (80, 120),  (50, 100), (10, 20), (80, 150), "cucurbit", 0, "medium", "short", (18, 30), (500, 800),  "Summer"),
    ("Pumpkins",   (6.0, 7.0), (80, 120),  (50, 100), (10, 20), (100, 150), "cucurbit", 0, "deep", "short",  (18, 30), (500, 800),  "Summer"),
    ("Cantaloupe", (6.0, 7.0), (80, 130),  (50, 100), (10, 20), (80, 150), "cucurbit", 0, "deep",  "short",  (20, 32), (400, 700),  "Summer"),
    ("Bell Peppers", (6.0, 6.8), (120, 180), (60, 120), (15, 25), (100, 200), "nightshade", 0, "medium", "medium", (18, 30), (600, 1200), "Spring,Summer"),
    ("Onions",     (6.0, 7.0), (100, 170), (60, 120), (20, 40), (80, 150), "allium",  0, "shallow", "short", (13, 27), (350, 600),  "Spring,Fall"),
    ("Garlic",     (6.0, 7.0), (100, 160), (50, 100), (20, 40), (80, 140), "allium",  0, "shallow", "short", (12, 24), (400, 700),  "Fall,Spring"),
    ("Carrots",    (6.0, 6.8), (70, 120),  (50, 100), (10, 20), (100, 200), "carrot", 0, "deep",   "short",  (15, 24), (400, 700),  "Spring,Fall"),
    ("Celery",     (6.0, 7.0), (150, 250), (80, 150), (15, 30), (150, 250), "carrot", 0, "shallow", "medium", (15, 24), (600, 1000), "Spring,Fall"),
    ("Radishes",   (6.0, 7.0), (50, 80),   (40, 80), (15, 25), (60, 120), "brassica", 0, "shallow", "short", (10, 22), (300, 600),  "Spring,Fall"),
    ("Turnips",    (6.0, 7.0), (60, 100),  (40, 80), (15, 25), (60, 120), "brassica", 0, "medium", "short",  (10, 22), (350, 650),  "Spring,Fall"),
    ("Brussels Sprouts", (6.0, 7.0), (150, 200), (60, 100), (25, 40), (100, 180), "brassica", 0, "medium", "medium", (12, 22), (600, 900), "Fall"),
    ("Okra",       (6.0, 7.0), (80, 120),  (40, 80), (10, 20), (60, 120), "mallow",  0, "deep",   "tall",   (22, 34), (600, 1000), "Summer"),
    ("Sweet Potatoes", (5.5, 6.5), (40, 80), (40, 80), (10, 20), (100, 180), "morning glory", 0, "deep", "short", (22, 32), (500, 900), "Summer"),
]



# Secondary + micronutrient demand / response: L low, M medium, H high.
# Source: Michigan State University Extension E-486 "Secondary and Micronutrients for Vegetables and Field Crops"
# (Vitosh, Warncke & Lucas), plus Ca/Mg/S needs from standard extension guides (e.g. Ca: blossom-end rot in tomato
# and pepper, pod fill in peanut, tipburn in lettuce/cabbage; S: brassicas and alfalfa; Mg: potato, tomato, grape, beet).
#                    S    Ca   Mg   Zn   Fe   Mn   Cu   B    Mo
MICRO = {
    "Arugula":        "H    M    M    L    M    M    L    H    H",
    "Asparagus":      "L    M    M    L    M    L    L    L    L",
    "Beet":           "M    M    H    M    H    H    H    H    H",
    "Broccoli":       "H    H    M    L    H    M    M    H    H",
    "Cabbage":        "H    H    M    L    M    M    M    M    M",
    "Cauliflowers":   "H    H    M    L    H    M    M    H    H",
    "Chard":          "M    M    H    M    H    H    H    M    H",
    "Chilli Peppers": "M    H    H    M    M    M    L    M    L",
    "Cress":          "H    M    L    L    M    M    L    M    M",
    "Cucumbers":      "L    M    M    L    L    H    M    L    L",
    "Eggplants":      "M    H    H    M    M    M    M    M    M",
    "Endive":         "M    H    M    M    M    H    H    M    H",
    "Grapes":         "L    M    H    H    H    M    L    H    L",
    "Green Peas":     "M    M    M    L    M    H    L    L    M",
    "Kale":           "H    H    M    L    M    M    M    H    M",
    "Lettuce":        "M    H    M    M    M    H    H    M    H",
    "Potatoes":       "M    H    H    M    L    H    L    L    L",
    "Radicchio":      "M    H    M    M    M    H    H    M    H",
    "Spinach":        "M    M    H    H    H    H    H    M    H",
    "Strawberry":     "L    H    M    M    M    M    L    H    L",
    "Tomatoes":       "M    H    H    M    H    M    H    M    M",
    "Watermelon":     "L    H    M    L    L    M    M    M    M",
    "Coffee":         "M    M    H    H    M    M    M    H    L",
    "Corn":           "M    L    M    H    M    M    M    L    L",
    "Cotton":         "M    M    M    M    M    M    L    H    L",
    "Hay":            "H    H    M    L    M    M    H    H    M",
    "Peanuts":        "M    H    M    M    M    M    L    H    M",
    "Rice":           "M    L    M    H    H    M    M    L    L",
    "Soybeans":       "M    M    M    M    H    H    L    L    M",
    "Sugar Beets":    "M    M    H    M    H    M    M    H    M",
    "Sugarcane":      "M    M    M    M    M    M    M    L    L",
    "Wheat":          "M    L    L    L    L    H    H    L    L",
    "Oats":           "M    L    L    L    M    H    H    L    L",
    "Barley":         "M    L    L    M    M    H    H    L    L",
    "Rye":            "L    L    L    L    L    L    L    L    L",
    "Triticale":      "M    L    L    L    L    H    H    L    L",
    "Sorghum":        "M    L    M    H    H    H    M    L    L",
    "Millet":         "L    L    L    M    M    M    L    L    L",
    "Sweet Corn":     "M    L    M    H    M    M    M    M    L",
    "Sunflowers":     "M    M    M    M    M    M    M    H    M",
    "Safflower":      "M    M    M    M    M    M    L    M    L",
    "Canola":         "H    M    M    M    M    M    L    H    M",
    "Mustard":        "H    M    M    L    M    M    L    H    M",
    "Flax":           "M    M    M    H    M    M    M    L    L",
    "Buckwheat":      "L    M    M    L    M    M    L    L    L",
    "Dry Beans":      "M    M    M    H    H    H    L    L    M",
    "Dry Peas":       "M    M    M    L    M    H    L    L    M",
    "Lentils":        "M    M    M    M    M    M    L    L    M",
    "Chickpeas":      "M    M    M    M    H    M    L    L    M",
    "Snap Beans":     "M    M    M    H    H    H    L    L    M",
    "Squash":         "L    H    M    L    M    M    M    M    L",
    "Pumpkins":       "L    H    M    L    M    M    M    M    L",
    "Cantaloupe":     "L    H    M    L    M    M    M    M    M",
    "Bell Peppers":   "M    H    H    M    M    M    L    M    L",
    "Onions":         "H    M    M    H    M    H    H    L    H",
    "Garlic":         "H    M    M    M    M    H    M    L    M",
    "Carrots":        "L    M    M    L    M    M    M    M    L",
    "Celery":         "M    H    H    M    M    M    M    H    L",
    "Radishes":       "H    M    M    M    M    M    M    M    M",
    "Turnips":        "H    M    M    L    M    M    M    H    M",
    "Brussels Sprouts": "H  H    M    L    M    M    M    H    H",
    "Okra":           "M    H    M    M    M    M    L    M    L",
    "Sweet Potatoes": "M    M    H    M    M    M    L    M    L",
}
MICRO_NUTS = ["S", "Ca", "Mg", "Zn", "Fe", "Mn", "Cu", "B", "Mo"]
WORD = {"L": "low", "M": "medium", "H": "high"}


def main():
    d = pd.read_csv(SRC)
    q = lambda s, p: s.quantile(p)
    rows = []
    for name, g in d.groupby("Name"):
        fam, fix, roots, height = BOTANY[name]
        rows.append(dict(
            name=name, source="Kaggle snmahsa/soil-nutrients (simulated profiles)",
            ph_min=round(q(g.pH, .05), 1), ph_max=round(q(g.pH, .95), 1),
            N_kg_ha=round(g.Nitrogen.median()), P_kg_ha=round(g.Phosphorus.median()), K_kg_ha=round(g.Potassium.median()), S_kg_ha=None,
            temp_min_c=round(q(g.Temperature, .05), 1), temp_max_c=round(q(g.Temperature, .95), 1),
            rain_min_mm=round(q(g.Rainfall, .05)), rain_max_mm=round(q(g.Rainfall, .95)),
            humidity_pct=round(g.Rh.median()), light_hours=round(g.Light_Hours.median(), 1),
            soil_types=",".join(g.Soil_Type.value_counts().index), seasons=",".join(g.Season.value_counts().index),
            photoperiod=g.Photoperiod.mode()[0], fertility_need=g.Fertility.mode()[0],
            typical_yield_t_ha=round(g.Yield.median(), 1),
            family=fam, fixes_n=fix, roots=roots, height=height))
    kaggle = {r["name"] for r in rows}
    for (name, ph, n, p, s, k, fam, fix, roots, height, t, rain, seasons) in FIELD:
        rows.append(dict(name=name, source="Team research doc (OSU, Texas A&M, LSU, MSU) + FAO crop sheets",
                         ph_min=ph[0], ph_max=ph[1], N_kg_ha=sum(n) / 2, P_kg_ha=sum(p) / 2, K_kg_ha=sum(k) / 2, S_kg_ha=sum(s) / 2,
                         temp_min_c=t[0], temp_max_c=t[1], rain_min_mm=rain[0], rain_max_mm=rain[1], soil_types="Loam,Clay Loam,Silt Loam",
                         seasons=seasons, family=fam, fixes_n=fix, roots=roots, height=height))
    df = pd.DataFrame(rows)
    # hunger level for N, P, K: compared across ALL plants in the table (bottom third = low, top third = high)
    for nut in ("N", "P", "K"):
        col = f"{nut}_kg_ha"
        lo, hi = df[col].quantile([1 / 3, 2 / 3])
        df[f"{nut}_hunger"] = df[col].apply(lambda v: "low" if v < lo else "high" if v > hi else "medium")
    df.loc[df.fixes_n == 1, "N_hunger"] = "fixes own N"
    for i, nut in enumerate(MICRO_NUTS):
        df[f"{nut}_hunger"] = df.name.map(lambda n: WORD[MICRO[n].split()[i]])
    df = df.sort_values(["source", "name"]).reset_index(drop=True)
    df.to_csv(HERE / "plants.csv", index=False)
    print(f"wrote plants.csv: {len(df)} plants ({len(kaggle)} from Kaggle, {len(FIELD)} field crops)")
    short = {"low": "L", "medium": "M", "high": "H", "fixes own N": "fix"}
    show = df[["name", "ph_min", "ph_max"]].copy()
    for nut in ["N", "P", "K"] + MICRO_NUTS:
        show[nut] = df[f"{nut}_hunger"].map(short)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
