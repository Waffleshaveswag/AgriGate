"""
train_lightgbm.py - baseline LightGBM models on county_crop_train.csv, tested on states the model never saw.

  Task A  grown       classification: will this crop be grown in this county?   metric: ROC-AUC
  Task B  yield_rel   regression: how good is the yield vs the national typical? metric: R², MAE

Spatial cross-validation: GroupKFold by STATE (5 folds), so neighbouring counties of the same state never sit in both
train and test. This is a much harder, more honest test than a random split.
Baselines: the hand-written rule tree (rule_score) for Task A; predicting each crop's average for Task B.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
meta = json.load(open(HERE / "feature_columns.json"))
df = pd.read_csv(HERE / "county_crop_train.csv", low_memory=False)
FEAT, CAT = meta["features"], meta["categorical"]
for c in CAT:
    df[c] = df[c].astype("category")
groups = df["state"].astype(str)
gkf = GroupKFold(n_splits=5)
PARAMS = dict(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

# ---------------------------------------------------------------- Task A: grown (classification)
oof = np.zeros(len(df))
for tr, te in gkf.split(df, df.grown, groups):
    m = lgb.LGBMClassifier(**PARAMS).fit(df.iloc[tr][FEAT], df.grown.iloc[tr], categorical_feature=CAT)
    oof[te] = m.predict_proba(df.iloc[te][FEAT])[:, 1]
auc = roc_auc_score(df.grown, oof)
auc_rule = roc_auc_score(df.grown, df.rule_score)
per_crop = df.assign(p=oof).groupby("crop", observed=True).apply(
    lambda g: roc_auc_score(g.grown, g.p) if g.grown.nunique() == 2 else np.nan, include_groups=False).sort_values()
print(f"TASK A  grown   ROC-AUC on unseen states: LightGBM {auc:.3f}   vs rule tree alone {auc_rule:.3f}")
print("  per crop:", ", ".join(f"{k} {v:.2f}" for k, v in per_crop.items()))

# ---------------------------------------------------------------- Task B: yield_rel (regression)
y = df.dropna(subset=["yield_rel"]).reset_index(drop=True)
oofy = np.zeros(len(y))
for tr, te in gkf.split(y, y.yield_rel, y["state"].astype(str)):
    m = lgb.LGBMRegressor(**PARAMS).fit(y.iloc[tr][FEAT], y.yield_rel.iloc[tr], categorical_feature=CAT)
    oofy[te] = m.predict(y.iloc[te][FEAT])
base = y.groupby("crop", observed=True).yield_rel.transform("mean")
print(f"TASK B  yield_rel  on unseen states: LightGBM R² {r2_score(y.yield_rel, oofy):.3f}, MAE {mean_absolute_error(y.yield_rel, oofy):.3f}"
      f"   vs crop-average baseline R² {r2_score(y.yield_rel, base):.3f}, MAE {mean_absolute_error(y.yield_rel, base):.3f}")

# ---------------------------------------------------------------- final models on all data + importance
final_a = lgb.LGBMClassifier(**PARAMS).fit(df[FEAT], df.grown, categorical_feature=CAT)
final_b = lgb.LGBMRegressor(**PARAMS).fit(y[FEAT], y.yield_rel, categorical_feature=CAT)
final_a.booster_.save_model(str(HERE / "model_grown.txt"))
final_b.booster_.save_model(str(HERE / "model_yield_rel.txt"))
imp = pd.Series(final_a.booster_.feature_importance("gain"), index=FEAT).sort_values(ascending=False)
print("\nTop features (grown, by gain):", ", ".join(f"{k}" for k in imp.head(15).index))
impb = pd.Series(final_b.booster_.feature_importance("gain"), index=FEAT).sort_values(ascending=False)
print("Top features (yield_rel, by gain):", ", ".join(f"{k}" for k in impb.head(15).index))
json.dump(dict(grown_auc=round(auc, 4), grown_auc_rule_tree=round(auc_rule, 4), per_crop_auc=per_crop.round(3).to_dict(),
               yield_r2=round(r2_score(y.yield_rel, oofy), 4), yield_mae=round(mean_absolute_error(y.yield_rel, oofy), 4),
               yield_r2_baseline=round(r2_score(y.yield_rel, base), 4), cv="GroupKFold(5) by state",
               top_features_grown=imp.head(20).round(1).to_dict(), top_features_yield=impb.head(20).round(1).to_dict()),
          open(HERE / "baseline_results.json", "w"), indent=1)
print("\nsaved model_grown.txt, model_yield_rel.txt, baseline_results.json")
