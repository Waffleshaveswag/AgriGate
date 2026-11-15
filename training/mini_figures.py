"""Small, big-text panels for a 2-slide summary (each shown about 380 x 260 px on a slide). Real data and model.

figures/mini_nasa.png    NASA POWER summer rain per county
figures/mini_corn.png    corn: USDA real map vs model map (state held out)
figures/mini_2050.png    Des Moines chance today vs 2050 (corn, soybeans, wheat)
figures/mini_boost.png   one prediction built up tree by tree (Iowa corn vs Iowa cotton)
figures/mini_fair.png    the 5 groups of states used for the fair test
figures/mini_score.png   coin flip vs hand-written rules vs LightGBM
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent; FIG = HERE / "figures"
BLUE, ORANGE, TEAL, PURPLE, MAGENTA, GREEN, GREY, DARK = "#0f62fe", "#eb6200", "#009d9a", "#8a3ffc", "#d02670", "#198038", "#8d8d8d", "#161616"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 14})
SIZE = (3.8, 2.6)
meta = json.load(open(HERE / "feature_columns.json")); FEAT, CAT = meta["features"], meta["categorical"]
res = json.load(open(HERE / "final_results.json"))
m = lgb.Booster(model_file=str(HERE / "models" / "lightgbm_grown_tuned.txt"))
CATS = dict(zip(CAT, m.pandas_categorical))
data = pd.read_parquet(HERE / "county_crop_train.parquet")
for c in CAT: data[c] = data[c].astype(pd.CategoricalDtype(CATS[c]))
counties = data.drop_duplicates("fips")

def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.04, facecolor="white"); plt.close(fig); print("wrote", name)

def usmap(ax):
    ax.set_aspect(1.25); ax.axis("off")

# NASA summer rain
fig, ax = plt.subplots(figsize=SIZE)
sc = ax.scatter(counties.lon, counties.lat, c=counties.clim_summer_rain_mm, cmap="YlGnBu", s=1.6)
usmap(ax); ax.text(-125, 50.5, "NASA summer rain", fontsize=15, weight="bold")
ax.text(-125, 23.2, "dry", fontsize=13, color="#8a8a3a"); ax.text(-72, 23.2, "wet", fontsize=13, color="#1d3f8f", ha="right")
save(fig, "mini_nasa.png")

# corn: real vs model (out-of-fold predictions, state held out)
A = data.dropna(subset=["grown_100ac"]).reset_index(drop=True)
params = dict(res["params_tuned"], objective="binary", verbose=-1)
corn_pred = pd.Series(np.nan, index=A.index)
for tr, te in GroupKFold(5).split(A, A.grown_100ac, A.state.astype(str)):
    mk = lgb.train(params, lgb.Dataset(A.iloc[tr][FEAT], A.grown_100ac.iloc[tr], categorical_feature=CAT), num_boost_round=res["boosting_rounds"])
    te_c = [i for i in te if str(A.crop.iloc[i]) == "Corn"]
    corn_pred.iloc[te_c] = mk.predict(A.iloc[te_c][FEAT])
C = A[A.crop.astype(str) == "Corn"].copy(); C["pred"] = corn_pred[C.index]
fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE)
a1.scatter(C.lon, C.lat, c=C.grown_100ac, cmap="Greens", vmin=-0.3, vmax=1, s=0.9); usmap(a1)
a2.scatter(C.lon, C.lat, c=C.pred, cmap="Greens", vmin=-0.3, vmax=1, s=0.9); usmap(a2)
a1.set_title("Real (USDA)", fontsize=14, weight="bold"); a2.set_title("Model's guess", fontsize=14, weight="bold")
fig.subplots_adjust(wspace=0.05)
save(fig, "mini_corn.png")

# 2050
sys.path.insert(0, "/home/user/AgriGate_LightGBM/model")
import agrigate_model as AG
fu = AG.future(41.6, -93.6, 2050); byc = {r["crop"]: r for r in fu["crops"]}
show = ["Corn", "Soybeans", "Wheat"]
fig, ax = plt.subplots(figsize=SIZE)
xs = np.arange(3); w = 0.38
now = [byc[c]["chance_now"] * 100 for c in show]; fut = [byc[c]["chance_future"] * 100 for c in show]
ax.bar(xs - w / 2, now, w, color="#a6c8ff", label="today"); ax.bar(xs + w / 2, fut, w, color=BLUE, label="2050")
for i in range(3):
    ax.text(xs[i] - w / 2, now[i] + 2, f"{now[i]:.0f}", ha="center", fontsize=12)
    ax.text(xs[i] + w / 2, fut[i] + 2, f"{fut[i]:.0f}", ha="center", fontsize=12, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels(show, fontsize=14); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=12, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
ax.set_title("Des Moines: chance %", fontsize=14, weight="bold", loc="left")
save(fig, "mini_2050.png")

# boosting
N = m.num_trees(); ks = np.unique(np.r_[1:21, np.geomspace(1, N, 60).astype(int), N])
fig, ax = plt.subplots(figsize=SIZE)
for crop, col, lab in [("Corn", GREEN, "Iowa corn"), ("Cotton", MAGENTA, "Iowa cotton")]:
    X = data[(data.fips == "19153") & (data.crop.astype(str) == crop)][FEAT]
    pr = [float(m.predict(X, num_iteration=int(k))[0]) * 100 for k in ks]
    ax.plot(ks, pr, color=col, lw=3)
    ax.text(N * 1.12, pr[-1], f"{pr[-1]:.0f}%", va="center", fontsize=15, weight="bold", color=col)
    ax.text(1.1, pr[0] + (7 if crop == "Corn" else -11), lab, fontsize=13, color=col, weight="bold")
ax.set_xscale("log"); ax.set_xlim(0.9, N * 2.4); ax.set_ylim(-8, 108)
ax.set_xticks([1, 10, 100, N]); ax.set_xticklabels(["1", "10", "100", str(N)])
ax.set_yticks([0, 50, 100]); ax.set_xlabel("trees")
save(fig, "mini_boost.png")

# fair test map
fold_of_state = {}
for k, (tr, te) in enumerate(GroupKFold(5).split(A, A.grown_100ac, A.state.astype(str))):
    for s in A.state.astype(str).iloc[te].unique(): fold_of_state[s] = k
cc = counties.copy(); cc["fold"] = cc.state.astype(str).map(fold_of_state)
fig, ax = plt.subplots(figsize=SIZE)
for k, col in enumerate([BLUE, ORANGE, TEAL, PURPLE, MAGENTA]):
    d = cc[cc.fold == k]; ax.scatter(d.lon, d.lat, s=1.6, c=col)
usmap(ax)
save(fig, "mini_fair.png")

# score
pc = res["grown_auc_median_per_crop"]
rows = [("Coin flip", 0.50, GREY), ("Our rules", pc["rule_tree"], ORANGE), ("LightGBM", pc["lightgbm_tuned"], BLUE)]
fig, ax = plt.subplots(figsize=SIZE)
for i, (lab, v, col) in enumerate(rows[::-1]):
    ax.barh(i, v, color=col, height=0.6); ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=16, weight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=14)
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.axvline(0.5, color=GREY, ls="--", lw=1)
save(fig, "mini_score.png")
