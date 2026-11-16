# AgriVision LightGBM training dataset

**`county_crop_train.csv` / `.parquet`**: 177,156 rows = **3,108 US counties × 57 crops**, **94 features** and **real USDA labels**. It's built for gradient boosting (LightGBM, XGBoost, CatBoost). The text columns are already typed as categories in the parquet file.

## The question it answers
> "Given the soil, nutrients and climate of this county, and this crop's needs: is the crop grown here commercially, and how good is its yield?"

## Labels (targets), from real USDA observations
| Column | Type | Meaning |
|---|---|---|
| `grown` | 0/1 | The 2017 or 2022 Census of Agriculture reports the crop harvested in the county ("(D)", withheld for privacy, still counts as grown). Includes tiny plantings. |
| `grown_100ac` | 0/1/NaN | **Commercial presence:** at least 100 acres harvested. NaN where the census withheld the acreage. **This is the recommended classification target.** |
| `acres`, `area_share` | number | Mean harvested acres (2017, 2022), and that area ÷ the county's land area |
| `yield`, `yield_unit` | number | Mean NASS survey yield 2014–2023 in the crop's own unit (bu/ac, lb/ac, t/ac, cwt/ac). 11 field crops. |
| `yield_rel` | number | `yield` ÷ the national median county yield of that crop. 1.0 = typical. **Recommended regression target:** it puts all crops on one scale. |

## Features (94)
| Prefix | Source | What |
|---|---|---|
| `soil_*` | **USDA NRCS SSURGO** (Soil Data Access), area-weighted surface horizon per county; 77 % of counties. The rest are filled from FAO HWSD values at nearby sites. | pH, organic matter, CEC, water holding (mm/m), sand/silt/clay, lime, salinity (EC), erodibility K, slope, share poorly drained, estimated N |
| `usgs_*` | **USGS OFR 2014-1082** soil chemistry, the 5 nearest sites, distance-weighted | P, K, S, Ca, Mg, Zn, Fe, Mn, Cu, Mo (mg/kg), organic carbon |
| `clim_*` | **NASA POWER** climatology 2001–2020 | rain, PET, aridity, mean / warmest / coldest month temperature, dry months, months above 10 °C, root-zone wetness, May–Sep rain |
| `crop_*` | `plants.csv` (team research doc, Kaggle profiles, MSU E-486) | pH range, need 0–3 for N P K S Ca Mg Zn Fe Mn Cu B Mo, N fixing, family, roots, temperature and rain range, perennial, salt sensitivity |
| `soil_level_*` | derived | Each nutrient's level 1–3 in the county, lowered when pH locks it up |
| `m_*` | derived **match features** | pH gap, pH distance from the crop's ideal, temperature gap, rain ratio, salt risk, need − supply for 11 nutrients, count of nutrient shortfalls |
| `rule_*` | the hand-written decision tree | Its pass/fail verdict and score, as features |

`feature_columns.json` lists the feature, categorical, label and ID columns, plus the grouping column for cross-validation (`state`).

**Don't use as features:** `fips`, `county`, `lat`, `lon`, `land_acres` (IDs) and every label column.

## Baseline results: `train_lightgbm.py`, 5-fold cross-validation grouped **by state** (tested on states never seen in training)
| Task | LightGBM | Baseline |
|---|---|---|
| `grown` (any planting) | ROC-AUC **0.888** | rule tree alone 0.499 (coin flip) |
| `grown_100ac` (commercial) | ROC-AUC 0.976 overall; **median per-crop 0.75**: peanuts 0.95, cotton 0.94, soybeans 0.93, sugarcane 0.93, sugar beets 0.93, rice 0.90, corn 0.85, wheat 0.81 | rule tree 0.56 |
| `yield_rel` | **R² 0.36**, MAE 0.143 | crop average R² 0.00, MAE 0.192 |

The overall AUC is inflated by crop identity, because some crops are almost always or almost never grown. **Report the per-crop AUC.** Most predictive: crop, summer rain, summer heat, salinity, root-zone wetness, drainage, aridity. For yield: soil P, aridity, summer rain and heat, drainage, pH.

## Known limits
- **County level, not field level.** Soil and climate are county averages; the labels are county totals.
- **Weak for specialty vegetables.** Broccoli, lettuce, endive and tomatoes are grown commercially in only a few states, with California dominating. When a whole state is held out, there's little left to learn from.
- **Nutrients are total amounts from survey sites** (not lab fertilizer numbers), and N is estimated from organic matter.
- **"Not grown" ≠ "can't grow".** Markets, irrigation, farm size and history decide too. The model learns where crops **are** grown, which mixes suitability with economics.
- **Missing crops:** strawberries, grapes, chard, arugula, radicchio and coffee aren't in the census harvested-area table, so they have no labels.

## Rebuild
```
python3 build_training_data.py     # uses raw/ + ../soil_dataset (soil_sites.csv, plants.csv) + SSURGO (cached)
python3 train_lightgbm.py          # baseline models -> model_grown.txt, model_yield_rel.txt, baseline_results.json
```


## Tuned model (`tune_and_figures.py`)
**Step 1, data:** NASA POWER climate + USDA SSURGO soil + USGS nutrients, with USDA NASS as the answers.
**Step 2, prepared and adjusted for LightGBM:** joined into 177,156 county × crop rows; gaps filled; pH-adjusted nutrient levels; 25 engineered match features.
**Step 3, tuned and trained LightGBM:** 12 settings searched with 3-fold cross-validation grouped by state; final 5-fold test on unseen states; early stopping at 497 trees.

| | Hand-written rules | LightGBM default | **LightGBM tuned** |
|---|---|---|---|
| Grown? median per-crop AUC | 0.58 | 0.82 | **0.81** |
| Yield R² (crop average = 0.00) | — | 0.36 | **0.36** |

Tuning picked the best of 12 settings on a quicker 3-fold test (0.84). On the full 5-fold test the default settings scored slightly higher (0.82 vs 0.81), so the gain from tuning did not hold up. The yield model scores the same either way.
Among location data, **NASA POWER climate carries 39 % of the model's importance**, SSURGO soil 26 %, USGS nutrients 17 %, engineered match features 19 %.
Tuned settings: `{'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.9, 'bagging_fraction': 1.0, 'bagging_freq': 1, 'lambda_l2': 20.0}`

Final models: `models/lightgbm_grown_tuned.txt`, `models/lightgbm_yield_tuned.txt`.

### Graphs (`figures/`)
| File | Shows |
|---|---|
| 01_pipeline.png | NASA data → prepare & adjust → tune & train LightGBM |
| 02_data_preparation.png | 94 features in 5 groups; size of the training table |
| 03_feature_patterns.png | Real patterns from NASA climate (corn vs summer rain, cotton vs heat, wheat vs aridity) |
| 04_tuning.png | The 12 LightGBM settings tried and the best one |
| 05_learning_curve.png | Accuracy on training vs unseen states as trees are added; early stopping |
| 06_results_vs_baselines.png | Rules vs LightGBM default vs tuned |
| 07_per_crop_accuracy.png | AUC for each crop |
| 08_yield_predicted_vs_actual.png | Predicted vs real USDA yields |
| 09_feature_importance_by_source.png | NASA climate is the biggest location factor |
| 10_map_corn_actual_vs_predicted.png | US map: real corn counties vs model prediction |
| 11_lightgbm_trees.png | Real trees from the model (tree 1, 2, 100, 497): the yes/no questions and leaf scores (`visualize_lightgbm.py`) |
| 12_lightgbm_boosting.png | Boosting: real predictions built up tree by tree, and how each new tree corrects less (`visualize_lightgbm.py`) |
