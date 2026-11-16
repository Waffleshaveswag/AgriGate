"""Visualize how the trained LightGBM model works, drawn from the real saved model (no retraining).

  figures/11_lightgbm_trees.png     - real trees from the model (tree 1, 2, 100, 446), top questions + leaf scores
  figures/12_lightgbm_boosting.png  - a real prediction built up tree by tree, and how each new tree corrects less
"""
import json, math, textwrap
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent; FIG = HERE / "figures"
GREEN, DARK, BLUE, GREY, SAND, BROWN, RED = "#3d7a54", "#173F32", "#2c78aa", "#8a948c", "#c9a24a", "#8c6840", "#b8402f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlelocation": "left"})
meta = json.load(open(HERE / "feature_columns.json")); FEAT, CAT = meta["features"], meta["categorical"]
m = lgb.Booster(model_file=str(HERE / "models" / "lightgbm_grown_tuned.txt"))
D = m.dump_model(); TREES = D["tree_info"]; FN = D["feature_names"]
CATS = dict(zip(CAT, m.pandas_categorical))           # category index -> name, per categorical feature
N_TREES = len(TREES)

NUTR = {"N": "nitrogen", "P": "phosphorus", "K": "potassium", "S": "sulfur", "Ca": "calcium", "Mg": "magnesium", "Zn": "zinc",
        "Fe": "iron", "Mn": "manganese", "Cu": "copper", "Mo": "molybdenum", "B": "boron"}
NAMES = {"clim_rain_mm": "NASA yearly rain (mm)", "clim_pet_mm": "NASA water loss to air (mm)", "clim_aridity": "NASA aridity (rain ÷ water loss)",
         "clim_t_mean": "NASA average temperature (°C)", "clim_t_warmest": "NASA warmest month (°C)", "clim_t_coldest": "NASA coldest month (°C)",
         "clim_dry_months": "NASA dry months", "clim_months_above_10c": "NASA months above 10 °C", "clim_root_wetness": "NASA root-zone wetness",
         "clim_summer_rain_mm": "NASA summer rain (mm)", "soil_ph": "Soil pH", "soil_om_pct": "Soil organic matter (%)",
         "soil_cec": "Soil nutrient-holding capacity", "soil_sand_pct": "Soil sand (%)", "soil_silt_pct": "Soil silt (%)", "soil_clay_pct": "Soil clay (%)",
         "soil_caco3_pct": "Soil lime (%)", "soil_ec_dsm": "Soil salinity", "soil_k_factor": "Soil erodibility", "soil_slope_pct": "Land slope (%)",
         "soil_poorly_drained_share": "Share of poorly drained soil", "soil_awc_mm_per_m": "Soil water-holding capacity",
         "nearest_site_km": "Distance to nearest USGS site (km)", "usgs_org_carbon_pct": "USGS organic carbon (%)", "soil_N_est_mgkg": "Estimated soil nitrogen",
         "crop": "Crop", "crop_family": "Crop family", "crop_ph_min": "Crop lowest pH", "crop_ph_max": "Crop highest pH",
         "crop_temp_min_c": "Crop lowest temperature (°C)", "crop_temp_max_c": "Crop highest temperature (°C)",
         "crop_rain_min_mm": "Crop minimum rain (mm)", "crop_rain_max_mm": "Crop maximum rain (mm)", "crop_fixes_n": "Crop fixes nitrogen",
         "crop_perennial": "Crop is perennial", "crop_salt_sensitive": "Crop is salt sensitive", "m_ph_gap": "pH outside crop's range",
         "m_ph_mid_diff": "pH distance from crop's ideal", "m_temp_gap": "Temperature outside crop's range", "m_rain_ratio": "Rain ÷ crop's minimum rain",
         "m_salt_risk": "Salt risk", "m_nutrient_shortfalls": "Number of nutrient shortfalls", "rule_soil_ok": "Hand-written rules say soil fits",
         "rule_score": "Hand-written rules score"}

def nice(f):
    if f in NAMES: return NAMES[f]
    p = f.split("_")
    if f.startswith("usgs_") and p[1] in NUTR: return f"USGS soil {NUTR[p[1]]} (mg/kg)"
    if f.startswith("crop_need_"): return f"Crop {NUTR[p[2]]} need"
    if f.startswith("crop_") and f.endswith("_kg_ha"): return f"Crop {NUTR[p[1]]} use (kg per hectare)"
    if f.startswith("soil_level_"): return f"Soil {NUTR[p[2]]} level"
    if f.startswith("m_need_minus_supply_"): return f"{NUTR[p[-1]].capitalize()} need minus supply"
    return f.replace("_", " ")

def question(node):
    f = FN[node["split_feature"]]
    if node["decision_type"] == "==":
        names = [CATS[f][int(i)] for i in str(node["threshold"]).split("||")]
        lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"{nice(f)} is {lst}?"
    t = node["threshold"]
    tt = f"{t:.0f}" if abs(t) >= 100 else f"{t:.1f}" if abs(t) >= 1 else f"{t:.2f}"
    return f"{nice(f)} ≤ {tt}?"

def gain_total(t):
    s, st = 0.0, [t["tree_structure"]]
    while st:
        n = st.pop()
        if "leaf_value" not in n: s += n["split_gain"]; st += [n["left_child"], n["right_child"]]
    return s

# ------------------------------------------------------------------ 11: real trees
def draw_tree(ax, tree, title, sub, depth=2):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.1, 9.85, title, fontsize=16, weight="bold", color=DARK, va="top")
    ax.text(0.1, 9.2, sub, fontsize=10.5, color="#555", va="top")
    def qtext(n, level):
        return "\n".join(textwrap.wrap(question(n), 26 if level == 0 else 23))
    def height(n, level):
        if "leaf_value" in n or level >= depth: return 1.0
        return 0.45 + 0.36 * qtext(n, level).count("\n") + 0.5
    def node_box(x, y, n, level, w):
        if "leaf_value" in n:
            v = n["leaf_value"]; c = GREEN if v > 0 else RED
            txt = f"Leaf: {v:+.3f}\n({n['leaf_count']:,} rows)"
            ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.5), w, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2", fc="white", ec=c, lw=2))
            ax.text(x, y, txt, ha="center", va="center", fontsize=9.5, color=c, weight="bold")
            return
        if level >= depth:
            ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.5), w, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2", fc="#eef1ee", ec=GREY, lw=1))
            ax.text(x, y, f"more questions\n({n['internal_count']:,} rows)", ha="center", va="center", fontsize=9, color="#555")
            return
        q = qtext(n, level)
        h = height(n, level)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    fc=BLUE if level == 0 else DARK, ec="none"))
        ax.text(x, y, q, ha="center", va="center", fontsize=9.6 if level == 0 else 8.8, color="white", weight="bold", linespacing=1.25)
        dx = 2.45 if level == 0 else 1.22
        kids = [n["left_child"], n["right_child"]]
        hk = max(height(c, level + 1) for c in kids)
        top_kid = y - h / 2 - (1.0 if level == 0 else 0.85)          # top edge of the child boxes
        ny = top_kid - hk / 2
        for side, sx, lab in (("left_child", x - dx, "yes"), ("right_child", x + dx, "no")):
            ax.annotate("", (sx, top_kid + 0.05), (x, y - h / 2), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GREY))
            ax.text((x + sx) / 2 + (-0.32 if lab == "yes" else 0.32), (y - h / 2 + top_kid) / 2 + 0.1, lab, ha="center", fontsize=9,
                    color=GREEN if lab == "yes" else RED, weight="bold")
            node_box(sx, top_kid - height(n[side], level + 1) / 2, n[side], level + 1, 3.6 if level == 0 else 2.2)
    node_box(5, 7.3, tree["tree_structure"], 0, 6.2)

show = [(0, "Tree 1"), (1, "Tree 2"), (99, "Tree 100"), (N_TREES - 1, f"Tree {N_TREES} (last)")]
fig = plt.figure(figsize=(20, 12.4))
fig.text(0.012, 0.985, f"Inside our LightGBM model: {N_TREES} decision trees, built one after another", fontsize=22, weight="bold", color=DARK, va="top")
fig.text(0.012, 0.947, "Real trees from the trained \"is this crop grown here?\" model. Each tree asks yes/no questions about the county and the crop, "
         "and ends in a leaf\nwith a score. A county's final score is the starting guess plus the leaf score from every tree; "
         "positive scores push towards \"grown\", negative towards \"not grown\".", fontsize=12.5, color="#444", va="top", linespacing=1.4)
for k, (ti, name) in enumerate(show):
    ax = fig.add_axes([0.012 + (k % 2) * 0.5, 0.465 - (k // 2) * 0.445, 0.475, 0.43])
    g = gain_total(TREES[ti]); leaves = TREES[ti]["num_leaves"]
    sub = f"{leaves} leaves · total improvement (gain) {g:,.0f}"
    if ti == 0: sub += " · also holds the starting guess"
    draw_tree(ax, TREES[ti], name, sub)
    if k in (0, 2):
        fig.text(0.5 - 0.006, 0.465 - (k // 2) * 0.445 + 0.2, "+", fontsize=40, weight="bold", color=DARK, ha="center", va="center")
fig.text(0.5 - 0.006, 0.255 + 0.02, "", fontsize=1)
fig.text(0.012, 0.012, "Only the top questions of each tree are drawn (each tree has up to 31 leaves). "
         "Early trees make big corrections (they ask which crop it is first); late trees make tiny ones.", fontsize=11, color="#666")
fig.savefig(FIG / "11_lightgbm_trees.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ 12: boosting, tree by tree
data = pd.read_parquet(HERE / "county_crop_train.parquet")
for c in CAT: data[c] = data[c].astype(pd.CategoricalDtype(CATS[c]))
examples = [("19153", "Corn", "Corn in Polk County, Iowa", GREEN), ("06019", "Cotton", "Cotton in Fresno County, California", SAND),
            ("19153", "Cotton", "Cotton in Polk County, Iowa", RED), ("06019", "Rice", "Rice in Fresno County, California", BLUE)]
ks = np.unique(np.r_[1:21, np.geomspace(1, N_TREES, 60).astype(int), N_TREES])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.8), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
for fips, crop, label, col in examples:
    row = data[(data.fips.astype(str).str.zfill(5) == fips) & (data.crop.astype(str) == crop)]
    if row.empty: continue
    X = row[FEAT]
    p = [float(m.predict(X, num_iteration=int(k))[0]) for k in ks]
    truth = row.grown_100ac.iloc[0]
    tl = "grown" if truth == 1 else "not grown" if truth == 0 else "unknown"
    a1.plot(ks, np.array(p) * 100, color=col, lw=2.6, label=f"{label}  (USDA: {tl})")
    a1.scatter([ks[-1]], [p[-1] * 100], color=col, s=45, zorder=3)
    a1.text(ks[-1] * 1.07, p[-1] * 100, f"{p[-1] * 100:.0f}%", va="center", color=col, weight="bold")
a1.set_xscale("log"); a1.set_xlim(0.8, N_TREES * 1.6); a1.set_ylim(-3, 106)
a1.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, N_TREES]); a1.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", str(N_TREES)])
a1.set_xlabel("number of trees added so far"); a1.set_ylabel("model's chance the crop is grown (%)")
a1.set_title("A real prediction, built up tree by tree")
a1.legend(loc="upper left", fontsize=10.5, frameon=False, bbox_to_anchor=(0.0, 1.0))
a1.annotate("after tree 1, every example is close\nto the starting guess (about 11–16%)", (1.05, 13), (1.12, 3), fontsize=10, color="#444",
            arrowprops=dict(arrowstyle="-", color=GREY))

g = np.array([gain_total(t) for t in TREES])
a2.fill_between(np.arange(1, N_TREES + 1), g, color=BLUE, alpha=0.25); a2.plot(np.arange(1, N_TREES + 1), g, color=BLUE, lw=1.8)
a2.set_yscale("log"); a2.set_xlim(1, N_TREES)
a2.set_xlabel("tree number"); a2.set_ylabel("how much the tree improves the model (gain, log scale)")
a2.set_title("Each new tree fixes less (boosting)")
a2.annotate(f"tree 1: {g[0]:,.0f}", (1, g[0]), (60, g[0] * 0.7), fontsize=10.5, arrowprops=dict(arrowstyle="-", color=GREY))
a2.annotate(f"tree {N_TREES}: {g[-1]:,.0f}", (N_TREES, g[-1]), (N_TREES * 0.55, g[-1] * 8), fontsize=10.5, arrowprops=dict(arrowstyle="-", color=GREY))
a2.text(0.98, 0.95, f"learning rate {float(json.load(open(HERE / 'final_results.json'))['params_tuned']['learning_rate']):g}: each tree's vote is shrunk\nso no single tree dominates; training stopped\nat tree {N_TREES} when "
        "states it never saw\nstopped getting better (early stopping)", transform=a2.transAxes, ha="right", va="top", fontsize=10.5, color="#444")
fig.suptitle("How boosting works: start from a guess, then each tree corrects the errors left by the trees before it",
             x=0.01, y=1.02, ha="left", fontsize=18, weight="bold", color=DARK)
fig.savefig(FIG / "12_lightgbm_boosting.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote 11_lightgbm_trees.png, 12_lightgbm_boosting.png")
