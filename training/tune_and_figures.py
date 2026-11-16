"""
tune_and_figures.py - fine-tune LightGBM (hyperparameter search, tested by state), train the final models, and draw
presentation figures that tell the story:  NASA data -> prepared for LightGBM -> trained LightGBM.

Outputs in figures/ (PNG, 16:9-friendly) + models/ + tuning_results.csv + final_results.json
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
MOD = HERE / "models"; MOD.mkdir(exist_ok=True)
GREEN, DARK, BLUE, RED, GREY, SAND = "#3d7a54", "#173F32", "#2c78aa", "#b8402f", "#8a948c", "#c9a24a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlelocation": "left"})

meta = json.load(open(HERE / "feature_columns.json"))
FEAT, CAT = meta["features"], meta["categorical"]
df = pd.read_parquet(HERE / "county_crop_train.parquet")
for c in CAT:
    df[c] = df[c].astype("category")
A = df.dropna(subset=["grown_100ac"]).reset_index(drop=True)
B = df.dropna(subset=["yield_rel"]).reset_index(drop=True)
GROUP = lambda d: d["state"].astype(str)

def per_crop_auc(d, p, y="grown_100ac"):
    s = d.assign(p=p).groupby("crop", observed=True).apply(lambda g: roc_auc_score(g[y], g.p) if g[y].nunique() == 2 else np.nan,
                                                             include_groups=False)
    return s.dropna().sort_values()

def cv_classifier(params, folds=5, rounds=2000, record=False):
    oof, curves, best_iters = np.zeros(len(A)), [], []
    for k, (tr, te) in enumerate(GroupKFold(folds).split(A, A.grown_100ac, GROUP(A))):
        ev = {}
        m = lgb.train({**params, "objective": "binary", "metric": "auc", "verbose": -1},
                      lgb.Dataset(A.iloc[tr][FEAT], A.grown_100ac.iloc[tr], categorical_feature=CAT),
                      num_boost_round=rounds,
                      valid_sets=[lgb.Dataset(A.iloc[tr][FEAT], A.grown_100ac.iloc[tr], categorical_feature=CAT),
                                  lgb.Dataset(A.iloc[te][FEAT], A.grown_100ac.iloc[te], categorical_feature=CAT)],
                      valid_names=["train", "unseen states"],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(ev)])
        oof[te] = m.predict(A.iloc[te][FEAT], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        if record and k == 0:
            curves = ev
    return oof, curves, int(np.median(best_iters))

# ---------------------------------------------------------------- 1. fine-tune (random search, grouped by state)
t0 = time.time()
DEFAULT = dict(learning_rate=0.04, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
rng = np.random.default_rng(7)
space = [DEFAULT] + [dict(learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])), num_leaves=int(rng.choice([15, 31, 63, 127])),
                          min_data_in_leaf=int(rng.choice([20, 50, 100, 200])), feature_fraction=float(rng.choice([0.5, 0.7, 0.9])),
                          bagging_fraction=float(rng.choice([0.7, 0.8, 1.0])), bagging_freq=1,
                          lambda_l2=float(rng.choice([0.0, 1.0, 5.0, 20.0]))) for _ in range(11)]
rows = []
for i, p in enumerate(space):
    oof, _, it = cv_classifier(p, folds=3)
    pc = per_crop_auc(A, oof)
    rows.append(dict(trial=i, **p, best_rounds=it, auc_overall=roc_auc_score(A.grown_100ac, oof), auc_median_crop=pc.median()))
    print(f"trial {i:2d}  median per-crop AUC {pc.median():.3f}  overall {rows[-1]['auc_overall']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
tune = pd.DataFrame(rows)
tune.to_csv(HERE / "tuning_results.csv", index=False)
best = tune.sort_values("auc_median_crop", ascending=False).iloc[0]
BEST = {k: best[k] for k in DEFAULT}
BEST["num_leaves"], BEST["min_data_in_leaf"], BEST["bagging_freq"] = int(BEST["num_leaves"]), int(BEST["min_data_in_leaf"]), 1
print("best params:", BEST)

# ---------------------------------------------------------------- 2. final 5-fold test of default vs tuned + learning curve
oof_def, _, _ = cv_classifier(DEFAULT)
oof_best, curves, best_rounds = cv_classifier(BEST, record=True)
pc_def, pc_best = per_crop_auc(A, oof_def), per_crop_auc(A, oof_best)
auc_rule = roc_auc_score(A.grown_100ac, A.rule_score)
pc_rule = per_crop_auc(A, A.rule_score.values)

oofy = np.zeros(len(B)); oofy_def = np.zeros(len(B))
for tr, te in GroupKFold(5).split(B, B.yield_rel, GROUP(B)):
    for params, out in ((BEST, oofy), (DEFAULT, oofy_def)):
        m = lgb.train({**params, "objective": "regression", "metric": "l2", "verbose": -1},
                      lgb.Dataset(B.iloc[tr][FEAT], B.yield_rel.iloc[tr], categorical_feature=CAT), num_boost_round=600)
        out[te] = m.predict(B.iloc[te][FEAT])
base_y = B.groupby("crop", observed=True).yield_rel.transform("mean")

# ---------------------------------------------------------------- 3. final models on all data
mA = lgb.train({**BEST, "objective": "binary", "verbose": -1}, lgb.Dataset(A[FEAT], A.grown_100ac, categorical_feature=CAT), num_boost_round=best_rounds)
mB = lgb.train({**BEST, "objective": "regression", "verbose": -1}, lgb.Dataset(B[FEAT], B.yield_rel, categorical_feature=CAT), num_boost_round=600)
mA.save_model(str(MOD / "lightgbm_grown_tuned.txt")); mB.save_model(str(MOD / "lightgbm_yield_tuned.txt"))

res = dict(params_tuned=BEST, boosting_rounds=best_rounds, cv="GroupKFold(5) by state (tested on unseen states)",
           grown_auc_overall=dict(rule_tree=auc_rule, lightgbm_default=roc_auc_score(A.grown_100ac, oof_def), lightgbm_tuned=roc_auc_score(A.grown_100ac, oof_best)),
           grown_auc_median_per_crop=dict(rule_tree=pc_rule.median(), lightgbm_default=pc_def.median(), lightgbm_tuned=pc_best.median()),
           grown_auc_per_crop_tuned=pc_best.round(3).to_dict(),
           yield_r2=dict(crop_average=r2_score(B.yield_rel, base_y), lightgbm_default=r2_score(B.yield_rel, oofy_def), lightgbm_tuned=r2_score(B.yield_rel, oofy)),
           yield_mae=dict(crop_average=mean_absolute_error(B.yield_rel, base_y), lightgbm_tuned=mean_absolute_error(B.yield_rel, oofy)))
json.dump(res, open(HERE / "final_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:1500])

# ---------------------------------------------------------------- 4. figures
def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig); print("figure", name)

# F1 pipeline
fig, ax = plt.subplots(figsize=(15, 6.2)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2)
def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18", fc=color, ec="none", alpha=0.95))
    ax.text(x + 0.18, y + h - 0.32, title, color="white", fontsize=13, weight="bold", va="top")
    ax.text(x + 0.18, y + h - 0.82, "\n".join(lines), color="white", fontsize=10.5, va="top", linespacing=1.45)
def arrow(x1, x2, y=3.1):
    ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=DARK))
box(0.1, 4.0, 3.6, 2.0, "NASA", ["POWER climate 2001-2020", "rain, heat, aridity, soil wetness", "(42-yr history for trends)"], BLUE)
box(0.1, 1.95, 3.6, 1.85, "USDA + USGS soil", ["SSURGO soil per county", "USGS: P K S Ca Mg Zn Fe Mn Cu Mo"], SAND)
box(0.1, 0.05, 3.6, 1.7, "USDA NASS (answers)", ["Census 2017/2022 harvested acres", "survey yields 2014-2023"], GREY)
arrow(3.85, 4.75)
box(4.8, 0.7, 4.3, 4.8, "1  Prepare + adjust data", [f"• join {df.fips.nunique():,} counties x {df.crop.nunique()} crops", f"  = {len(df):,} rows", "• fill gaps (SSURGO 77% + fallback)",
     "• pH-adjusted nutrient levels", "• 25 match features: pH gap,", "  need - supply x 11 nutrients,", "  temperature gap, salt risk",
     "• labels: grown 100+ acres,", "  yield vs national typical"], GREEN)
arrow(9.25, 10.15)
box(10.2, 0.7, 4.6, 4.8, "2  Tune + train LightGBM", [f"• {len(space)} settings searched", "• tested on states it never saw", "  (5-fold, grouped by state)",
     "• early stopping", f"• grown? AUC {pc_best.median():.2f} per crop", f"  (rules alone {pc_rule.median():.2f})", f"• yield R² {res['yield_r2']['lightgbm_tuned']:.2f}",
     "  (crop average 0.00)"], DARK)
ax.text(0.1, 6.15, "From NASA data to a trained crop model", fontsize=19, weight="bold", color=DARK, va="top")
save(fig, "01_pipeline.png")

# F2 data preparation: feature groups + dataset size
groups = {"NASA POWER climate": [f for f in FEAT if f.startswith("clim_")], "SSURGO soil": [f for f in FEAT if f.startswith("soil_") and not f.startswith("soil_level")],
          "USGS nutrients": [f for f in FEAT if f.startswith("usgs_") or f == "nearest_site_km"], "Crop needs": [f for f in FEAT if f.startswith("crop") ],
          "Engineered match features": [f for f in FEAT if f.startswith("m_") or f.startswith("soil_level") or f.startswith("rule_")]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
cols = [BLUE, SAND, "#8c6840", GREY, GREEN]
names = list(groups); counts = [len(groups[g]) for g in names]
a1.barh(names[::-1], counts[::-1], color=cols[::-1]); a1.set_title("Prepared for LightGBM: 94 features in 5 groups")
for i, v in enumerate(counts[::-1]): a1.text(v + 0.4, i, str(v), va="center", weight="bold")
a1.set_xlabel("number of features")
stats = [(f"{len(df):,}", "rows (county x crop)"), (f"{df.fips.nunique():,}", "US counties"), (str(df.crop.nunique()), "crops"), ("6", "public datasets joined"),
         (f"{int(A.grown_100ac.sum()):,}", "commercial crop-county pairs"), (f"{len(B):,}", "real yield records")]
a2.axis("off"); a2.set_title("The training table")
for i, (n, t) in enumerate(stats):
    y = 0.9 - i * 0.16
    a2.text(0.02, y, n, fontsize=24, weight="bold", color=DARK, transform=a2.transAxes, va="center")
    a2.text(0.42, y, t, fontsize=13, color="#333", transform=a2.transAxes, va="center")
save(fig, "02_data_preparation.png")

# F3 example engineered feature: summer rain vs commercial soybeans/cotton/corn
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, crop, feat, lab in ((axs[0], "Corn", "clim_summer_rain_mm", "May-Sep rain, mm (NASA POWER)"),
                            (axs[1], "Cotton", "clim_t_warmest", "warmest month, °C (NASA POWER)"),
                            (axs[2], "Potatoes", "m_ph_gap", "pH gap from crop's ideal range")):
    g = A[A.crop == crop]
    bins = pd.qcut(g[feat], 8, duplicates="drop")
    r = g.groupby(bins, observed=True).grown_100ac.mean()
    ax.plot([iv.mid for iv in r.index], r.values * 100, "-o", color=GREEN, lw=2.5)
    ax.set_title(crop); ax.set_xlabel(lab)
axs[0].set_ylabel("% of counties growing it commercially")
fig.suptitle("Why the features matter: real USDA patterns the model learns", x=0.01, ha="left", fontsize=16, weight="bold", y=1.03)
save(fig, "03_feature_patterns.png")

# F4 tuning
fig, ax = plt.subplots(figsize=(10, 5.2))
t = tune.sort_values("auc_median_crop")
colors = [RED if i == 0 else (GREEN if i == best.trial else GREY) for i in t.trial]
ax.barh([f"setting {i}" + (" (default)" if i == 0 else "") + (" ← best" if i == best.trial else "") for i in t.trial], t.auc_median_crop, color=colors)
ax.set_xlim(t.auc_median_crop.min() - 0.02, t.auc_median_crop.max() + 0.01)
ax.set_xlabel("median per-crop AUC on unseen states (3-fold, grouped by state)")
ax.set_title("Fine-tuning: searching LightGBM settings")
save(fig, "04_tuning.png")

# F5 learning curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(curves["train"]["auc"], color=GREY, lw=2, label="training states")
ax.plot(curves["unseen states"]["auc"], color=GREEN, lw=2.5, label="unseen states (test)")
bi = int(np.argmax(curves["unseen states"]["auc"]))
ax.axvline(bi, ls="--", color=DARK); ax.text(bi + 10, min(curves["unseen states"]["auc"]) + 0.002, f"early stop ~{bi} trees", color=DARK)
ax.set_xlabel("boosting rounds (trees)"); ax.set_ylabel("ROC-AUC"); ax.legend(frameon=False)
ax.set_title("Training LightGBM: accuracy as trees are added")
save(fig, "05_learning_curve.png")

# F6 results vs baselines
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
vals = [pc_rule.median(), pc_def.median(), pc_best.median()]
b = a1.bar(["Hand-written\nrules", "LightGBM\ndefault", "LightGBM\ntuned"], vals, color=[GREY, BLUE, GREEN])
a1.axhline(0.5, ls="--", color=RED); a1.text(2.45, 0.505, "coin flip", color=RED, ha="right")
a1.set_ylim(0.4, 1); a1.set_ylabel("median per-crop AUC"); a1.set_title("Is this crop grown here?")
for r, v in zip(b, vals): a1.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", weight="bold")
vals = [r2_score(B.yield_rel, base_y), r2_score(B.yield_rel, oofy_def), r2_score(B.yield_rel, oofy)]
b = a2.bar(["Crop\naverage", "LightGBM\ndefault", "LightGBM\ntuned"], vals, color=[GREY, BLUE, GREEN])
a2.set_ylim(0, max(vals) + 0.1); a2.set_ylabel("R² (1.0 = perfect)"); a2.set_title("How good is the yield?")
for r, v in zip(b, vals): a2.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", weight="bold")
fig.suptitle("Results on states the model never saw", x=0.01, ha="left", fontsize=16, weight="bold", y=1.03)
save(fig, "06_results_vs_baselines.png")

# F7 per-crop AUC
fig, ax = plt.subplots(figsize=(9, max(8.5, 0.3 * len(pc_best))))
ax.barh(pc_best.index, pc_best.values, color=[RED if v < 0.6 else (SAND if v < 0.75 else GREEN) for v in pc_best.values])
ax.axvline(0.5, ls="--", color=GREY); ax.set_xlim(0.3, 1.0)
for i, v in enumerate(pc_best.values): ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10)
ax.set_xlabel("ROC-AUC on unseen states"); ax.set_title("Tuned model accuracy for each crop")
save(fig, "07_per_crop_accuracy.png")

# F8 yield scatter
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(B.yield_rel, oofy, s=5, alpha=0.25, color=BLUE)
ax.plot([0.2, 1.8], [0.2, 1.8], "--", color=DARK); ax.set_xlim(0.2, 1.8); ax.set_ylim(0.2, 1.8)
ax.set_xlabel("actual yield ÷ national typical (USDA)"); ax.set_ylabel("predicted")
ax.set_title(f"Yield on unseen states (R² {r2_score(B.yield_rel, oofy):.2f})")
save(fig, "08_yield_predicted_vs_actual.png")

# F9 importance by data source + top features
imp = pd.Series(mA.feature_importance("gain"), index=FEAT)
src = {g: imp[fs].sum() for g, fs in groups.items()}
src = pd.Series(src).sort_values()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
a1.barh(src.index, src.values / src.sum() * 100, color=[{"NASA POWER climate": BLUE, "SSURGO soil": SAND, "USGS nutrients": "#8c6840",
                                                        "Crop needs": GREY, "Engineered match features": GREEN}[k] for k in src.index])
for i, v in enumerate(src.values / src.sum() * 100): a1.text(v + 0.5, i, f"{v:.0f}%", va="center", weight="bold")
a1.set_xlabel("% of what the model relies on (gain)"); a1.set_title("Which data the model uses")
top = (imp / imp.sum() * 100).sort_values().tail(12)
nice = lambda f: f.replace("clim_", "NASA ").replace("soil_", "soil ").replace("usgs_", "USGS ").replace("crop_", "crop ").replace("m_", "match ").replace("_", " ")
a2.barh([nice(f) for f in top.index], top.values, color=[BLUE if f.startswith("clim_") else GREEN if f.startswith(("m_", "rule")) else SAND if f.startswith("soil") else "#8c6840" if f.startswith("usgs") else GREY for f in top.index])
a2.set_xlabel("% importance"); a2.set_title("Top 12 features")
save(fig, "09_feature_importance_by_source.png")

# F10 maps: corn predicted vs actual
c = A[A.crop == "Corn"].copy(); c["p"] = oof_best[A.crop == "Corn"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
a1.scatter(c.lon, c.lat, c=c.grown_100ac, cmap="Greens", s=6, vmin=-0.3, vmax=1); a1.set_title("USDA: counties growing 100+ acres of corn")
s = a2.scatter(c.lon, c.lat, c=c.p, cmap="Greens", s=6, vmin=0, vmax=1); a2.set_title("LightGBM prediction (state held out)")
for a in (a1, a2): a.set_xticks([]); a.set_yticks([]); a.spines[:].set_visible(False)
fig.colorbar(s, ax=a2, shrink=0.8, label="chance grown")
save(fig, "10_map_corn_actual_vs_predicted.png")
print(f"done in {time.time() - t0:.0f}s")
