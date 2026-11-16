"""Simple, big-text figures for explaining AgriGate to a general (age 15+) audience. All drawn from the real data and model.

figures/kid_01_crop_maps.png      where corn, wheat, cotton and rice are really grown (USDA)
figures/kid_02_nasa_maps.png      NASA POWER average temperature and summer rain per county
figures/kid_03_soil_maps.png      soil pH (USDA SSURGO) and phosphorus (USGS) per county
figures/kid_04_report_card.png    coin flip vs hand-written rules vs LightGBM
figures/kid_05_desmoines_temp.png NASA yearly temperature at Des Moines + linear regression to 2050
figures/kid_06_crops_2050.png     model chance today vs 2050 for Des Moines crops
figures/kid_07_team_of_trees.png  one prediction built up tree by tree (corn and cotton in Iowa)
figures/kid_08_fair_test_map.png  the 5 groups of states used for the fair test (GroupKFold by state)
figures/kid_09_what_matters.png   share of importance by data source
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent; FIG = HERE / "figures"
BLUE, TEAL, PURPLE, MAGENTA, GREEN, ORANGE, GREY, DARK = "#0f62fe", "#009d9a", "#8a3ffc", "#d02670", "#198038", "#eb6200", "#8d8d8d", "#161616"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 19, "axes.titlelocation": "left",
                     "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14})
meta = json.load(open(HERE / "feature_columns.json")); FEAT, CAT = meta["features"], meta["categorical"]
res = json.load(open(HERE / "final_results.json"))
m = lgb.Booster(model_file=str(HERE / "models" / "lightgbm_grown_tuned.txt"))
CATS = dict(zip(CAT, m.pandas_categorical))
data = pd.read_parquet(HERE / "county_crop_train.parquet")
for c in CAT: data[c] = data[c].astype(pd.CategoricalDtype(CATS[c]))
counties = data.drop_duplicates("fips")

def us_axis(ax, title):
    ax.set_title(title, pad=10); ax.set_aspect(1.25); ax.axis("off")

def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)

# 1 ------------------------------------------------------------------ where crops are grown
fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
for ax, (crop, col) in zip(axs.flat, [("Corn", GREEN), ("Wheat", ORANGE), ("Cotton", PURPLE), ("Rice", BLUE)]):
    d = data[data.crop.astype(str) == crop]
    ax.scatter(d.lon, d.lat, s=5, c="#e0e0e0")
    g = d[d.grown_100ac == 1]
    ax.scatter(g.lon, g.lat, s=7, c=col)
    us_axis(ax, f"{crop}: {len(g):,} counties")
fig.suptitle("Where America really grows its crops (USDA farm records)", x=0.01, ha="left", fontsize=24, weight="bold", y=1.0)
fig.text(0.01, 0.005, "Each dot is a county. Coloured = at least 100 acres harvested (Census of Agriculture 2017/2022). Grey = not grown there.", fontsize=14, color="#525252")
save(fig, "kid_01_crop_maps.png")

# 2 ------------------------------------------------------------------ NASA climate maps
fig, axs = plt.subplots(1, 2, figsize=(17, 6.6))
for ax, col, cmap, title, unit in [(axs[0], "clim_t_mean", "RdYlBu_r", "Average temperature", "°C"),
                                   (axs[1], "clim_summer_rain_mm", "YlGnBu", "Summer rain (May–September)", "mm")]:
    sc = ax.scatter(counties.lon, counties.lat, c=counties[col], cmap=cmap, s=7)
    us_axis(ax, title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01); cb.set_label(unit)
fig.suptitle("What NASA POWER tells us about every county (2001–2020 average)", x=0.01, ha="left", fontsize=24, weight="bold", y=1.03)
save(fig, "kid_02_nasa_maps.png")

# 3 ------------------------------------------------------------------ soil maps
fig, axs = plt.subplots(1, 2, figsize=(17, 6.6))
sc = axs[0].scatter(counties.lon, counties.lat, c=counties.soil_ph, cmap="PuOr", vmin=4.5, vmax=8.5, s=7)
us_axis(axs[0], "Soil pH (USDA SSURGO)"); cb = fig.colorbar(sc, ax=axs[0], shrink=0.75, pad=0.01); cb.set_label("pH  (low = sour, high = chalky)")
p = counties.usgs_P_mgkg.clip(upper=counties.usgs_P_mgkg.quantile(0.97))
sc = axs[1].scatter(counties.lon, counties.lat, c=p, cmap="YlOrBr", s=7)
us_axis(axs[1], "Soil phosphorus (USGS)"); cb = fig.colorbar(sc, ax=axs[1], shrink=0.75, pad=0.01); cb.set_label("mg per kg of soil")
fig.suptitle("What is in the ground: soil acidity and a key plant nutrient", x=0.01, ha="left", fontsize=24, weight="bold", y=1.03)
save(fig, "kid_03_soil_maps.png")

# 4 ------------------------------------------------------------------ report card
pc = res["grown_auc_median_per_crop"]
rows = [("Flipping a coin", 0.50, GREY), ("Our hand-written rules", pc["rule_tree"], ORANGE),
        ("LightGBM (our trained model)", pc["lightgbm_tuned"], BLUE), ("A perfect model", 1.00, "#c6c6c6")]
fig, ax = plt.subplots(figsize=(14, 6))
for i, (lab, v, col) in enumerate(rows[::-1]):
    ax.barh(i, v, color=col, height=0.62)
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=22, weight="bold", color=DARK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=17)
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.axvline(0.5, color=GREY, ls="--", lw=1.2)
ax.set_xlabel("accuracy score (area under the curve), tested on states the model never saw")
ax.set_title("The report card: how often the model ranks a real crop county above a non-crop county", fontsize=18)
save(fig, "kid_04_report_card.png")

# 5 + 6 -------------------------------------------------------------- Des Moines climate + 2050 (real NASA POWER + model)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
import agrigate_model as AG
ct = AG.climate_trend(41.6, -93.6, 2050)
yrs = np.array(ct["years"]); t = ct["temperature"]
fig, ax = plt.subplots(figsize=(15, 6.6))
ax.plot(yrs, t["values"], "o-", color=ORANGE, lw=1.6, ms=6, label="NASA POWER: one dot per year")
x = np.array([yrs.min(), yrs.max()]); ax.plot(x, t["slope"] * x + t["intercept"], color=DARK, lw=3.2, label="linear regression (best straight line)")
x2 = np.array([yrs.max(), 2050]); ax.plot(x2, t["slope"] * x2 + t["intercept"], color=DARK, lw=3.2, ls="--", label="the same line extended to 2050")
ax.scatter([2050], [t["projected"]], s=120, color=DARK, zorder=3)
ax.annotate(f"2050: about {t['projected']:.1f} °C", (2050, t["projected"]), (2036, t["projected"] + 1.2), fontsize=18, weight="bold",
            arrowprops=dict(arrowstyle="-", color=GREY))
ax.text(1985, max(t["values"]) + 0.15, f"Warming {t['slope_per_decade']:+.2f} °C every 10 years   ·   R² = {t['r2']:.2f}", fontsize=17, color=BLUE, weight="bold")
ax.set_xlim(1982, 2054); ax.set_ylim(min(t["values"]) - 0.6, max(t["values"]) + 1.0)
ax.set_ylabel("average temperature for the year (°C)")
ax.set_title("Des Moines, Iowa: 42 years of NASA temperature data, and where the trend points")
ax.legend(loc="lower right", frameon=False, fontsize=14)
save(fig, "kid_05_desmoines_temp.png")

fu = AG.future(41.6, -93.6, 2050)
byc = {r["crop"]: r for r in fu["crops"]}
show = ["Corn", "Soybeans", "Wheat", "Hay", "Potatoes", "Tomatoes"]
fig, ax = plt.subplots(figsize=(15, 6.4))
xs = np.arange(len(show)); w = 0.38
now = [byc[c]["chance_now"] * 100 for c in show]; fut = [byc[c]["chance_future"] * 100 for c in show]
ax.bar(xs - w / 2, now, w, color="#78a9ff", label="today")
ax.bar(xs + w / 2, fut, w, color=BLUE, label="2050 (NASA trend)")
for i in range(len(show)):
    ax.text(xs[i] - w / 2, now[i] + 1.5, f"{now[i]:.0f}%", ha="center", fontsize=14)
    ax.text(xs[i] + w / 2, fut[i] + 1.5, f"{fut[i]:.0f}%", ha="center", fontsize=14, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["Hay (alfalfa)" if c == "Hay" else c for c in show], fontsize=16)
ax.set_ylim(0, 112); ax.set_ylabel("chance the crop is grown here (%)")
pr = fu["projection"]
ax.set_title(f"Des Moines in 2050: about {pr['warming_c_vs_2001_2020']:+.1f} °C and {pr['rain_change_pct']:+.0f}% rain. The model re-run says:")
ax.legend(frameon=False, fontsize=15, loc="upper right")
save(fig, "kid_06_crops_2050.png")

# 7 ------------------------------------------------------------------ team of trees
N = m.num_trees(); ks = np.unique(np.r_[1:21, np.geomspace(1, N, 70).astype(int), N])
fig, ax = plt.subplots(figsize=(15, 6.6))
for fips, crop, label, col in [("19153", "Corn", "Corn in Polk County, Iowa (USDA: grown)", GREEN), ("19153", "Cotton", "Cotton in Polk County, Iowa (USDA: not grown)", MAGENTA)]:
    X = data[(data.fips == fips) & (data.crop.astype(str) == crop)][FEAT]
    pr_ = [float(m.predict(X, num_iteration=int(k))[0]) * 100 for k in ks]
    ax.plot(ks, pr_, color=col, lw=4, label=label)
    ax.text(N * 1.08, pr_[-1], f"{pr_[-1]:.0f}%", va="center", fontsize=22, weight="bold", color=col)
ax.set_xscale("log"); ax.set_xlim(0.85, N * 1.7); ax.set_ylim(-4, 108)
ax.set_xticks([1, 3, 10, 30, 100, N]); ax.set_xticklabels(["1", "3", "10", "30", "100", str(N)])
ax.annotate("after 1 tree: just a rough guess", (1, 18), (1.3, 40), fontsize=15, arrowprops=dict(arrowstyle="-", color=GREY))
ax.set_xlabel("number of trees that have voted"); ax.set_ylabel("model's answer: chance it is grown (%)")
ax.set_title(f"A team of {N} trees: each one fixes a bit of what the others got wrong")
ax.legend(loc="center left", frameon=False, fontsize=15, bbox_to_anchor=(0, 0.62))
save(fig, "kid_07_team_of_trees.png")

# 8 ------------------------------------------------------------------ fair test map (real folds)
A = data.dropna(subset=["grown_100ac"]).reset_index(drop=True)
fold_of_state = {}
for k, (tr, te) in enumerate(GroupKFold(5).split(A, A.grown_100ac, A.state.astype(str))):
    for s in A.state.astype(str).iloc[te].unique(): fold_of_state[s] = k
cc = counties.copy(); cc["fold"] = cc.state.astype(str).map(fold_of_state)
cols = [BLUE, ORANGE, TEAL, PURPLE, MAGENTA]
fig, ax = plt.subplots(figsize=(15, 8.4))
for k in range(5):
    d = cc[cc.fold == k]; ax.scatter(d.lon, d.lat, s=9, c=cols[k], label=f"Group {k + 1}")
us_axis(ax, "The fair test: hide whole states, train on the others, then check the hidden ones")
ax.legend(loc="lower left", frameon=False, fontsize=16, markerscale=3, ncol=5, bbox_to_anchor=(0, -0.12))
fig.text(0.13, 0.04, "Each colour is one group of states. We train 5 times: each time one colour is hidden, and that colour is the exam.", fontsize=15, color="#525252")
save(fig, "kid_08_fair_test_map.png")

# 9 ------------------------------------------------------------------ what matters (donut)
imp = pd.Series(m.feature_importance("gain"), index=FEAT)
loc = imp[[f for f in FEAT if not f.startswith("crop")]]
groups = {"NASA climate": [f for f in loc.index if f.startswith("clim_")],
          "USDA soil": [f for f in loc.index if f.startswith("soil_") and not f.startswith("soil_level")],
          "USGS nutrients": [f for f in loc.index if f.startswith("usgs_") or f == "nearest_site_km"],
          "Match features we built": [f for f in loc.index if f.startswith(("m_", "rule_", "soil_level"))]}
share = pd.Series({g: loc[fs].sum() for g, fs in groups.items()}); share = share / share.sum() * 100
fig, ax = plt.subplots(figsize=(13, 7.2))
wedges, _ = ax.pie(share.values, colors=[BLUE, "#a56eff", ORANGE, TEAL], startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.38, edgecolor="white", linewidth=3))
ax.text(0, 0.06, f"{share['NASA climate']:.0f}%", ha="center", va="center", fontsize=46, weight="bold", color=BLUE)
ax.text(0, -0.2, "NASA climate", ha="center", va="center", fontsize=18, color=DARK)
ax.legend(wedges, [f"{g}: {v:.0f}%" for g, v in share.items()], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=18)
ax.set_title("What the model looks at most when it judges a place", fontsize=21)
save(fig, "kid_09_what_matters.png")
