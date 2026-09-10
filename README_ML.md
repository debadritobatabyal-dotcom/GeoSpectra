# Manganese Prospectivity Machine Learning Pipeline

This pipeline implements a single, authoritative `HistGradientBoostingClassifier` trained on 43 physically grounded and published geological features across the Central Indian Sausar Manganese Belt (20.95°N–22.15°N, 79.35°E–80.65°E). 

The production model bundle (`manganese_prospectivity_model.pkl`) encapsulates:
- Trained `HistGradientBoostingClassifier` estimator
- Deterministic categorical encoder mappings
- Empirically derived percentile thresholds (p80 Moderate, p95 High)
- Reference prediction distribution for continuous percentile rank scaling (0–100)
- `IsolationForest` applicability model for out-of-distribution environmental anomaly detection
- Sentinel-2 and Sentinel-1 baseline statistics

---

## 1. Data and Target Formulation

- **Training Dataset**: `data/dataset/sausar_manganese_core_features.csv` (20,000 regional stations).
- **Target Formulation**: Positive-Unlabeled (PU) learning on 18,883 filtered stations:
  - `POSITIVE = 1` (1,273 stations): Documented manganese ore horizons and confirmed deposits.
  - `UNLABELLED = 0` (17,610 stations): Regional background stations across the Sausar belt.
  - `UNCERTAIN` (1,117 stations): Ambiguous records strictly excluded from model fitting to prevent label noise.
- **Critical Distinction**: `UNLABELLED` denotes absence of published documentation, **not** confirmed absence of manganese. The model predicts relative prospectivity ranking rather than calibrated occurrence probabilities.

---

## 2. Authoritative 43-Feature Schema (v4.0.0)

Features are divided across 5 physical and geological domains:

1. **Spatial Coordinates (2 features)**:
   - `latitude`, `longitude`

2. **Sentinel-2 MSI Level-2A Surface Reflectance (10 features)**:
   - `B2` (Blue, 490 nm), `B3` (Green, 560 nm), `B4` (Red, 665 nm)
   - `B5` (Red Edge 1, 705 nm), `B6` (Red Edge 2, 740 nm), `B7` (Red Edge 3, 783 nm)
   - `B8` (NIR Broad, 842 nm), `B8A` (NIR Narrow, 865 nm)
   - `B11` (SWIR 1, 1610 nm), `B12` (SWIR 2, 2190 nm)

3. **Multispectral Ratios & Alteration Indices (11 features)**:
   - `NDVI` (Normalized Difference Vegetation Index)
   - `NDMI` (Normalized Difference Moisture Index)
   - `MNDWI` (Modified Normalized Difference Water Index)
   - `BSI` (Bare Soil Index)
   - `B4_B2_ratio` (Ferric Iron Oxide Index)
   - `B11_B12_ratio` (Silica / Carbonate Index)
   - `B12_B8_ratio` (Clay Alteration / Lithology Index)
   - `B11_B8_ratio` (Hydroxyl Alteration Index)
   - `gossan_alteration_index` (Gossan Mineral Index)
   - `red_edge_ratio_1`, `red_edge_ratio_2` (Red-edge spectral slopes)

4. **Sentinel-1 C-Band Synthetic Aperture Radar (5 features)**:
   - `VV`, `VH` (Backscatter in dB)
   - `VV_VH_ratio` (Cross-polarization ratio in dB)
   - `radar_backscatter_mean` (Mean SAR backscatter in dB)
   - `radar_texture` (Local SAR backscatter standard deviation)

5. **SRTM 30m DEM Geomorphometry (7 features)**:
   - `elevation_m` (Digital elevation, meters)
   - `slope_deg` (Slope angle, degrees)
   - `curvature` (Surface curvature)
   - `terrain_ruggedness` (Focal elevation standard deviation)
   - `local_relief_m` (Focal elevation range: max - min)
   - `topographic_position_index` (TPI: cell elevation - neighborhood mean)
   - `valley_or_ridge_class` *(Categorical)*: Landscape class (ridge, slope, valley)

6. **Geological Survey of India (GSI) Stratigraphy & Lithology (8 features)**:
   - `geological_group` *(Categorical)*: Stratigraphic group (e.g., Sausar Group, Tirodi Gneiss)
   - `geological_formation` *(Categorical)*: Formation (e.g., Mansar Formation, Chorbaoli Formation)
   - `stratigraphic_unit` *(Categorical)*: Member/Unit
   - `lithology` *(Categorical)*: Rock classification (e.g., Mica schist, Quartzite, Calc-silicate)
   - `metamorphic_grade` *(Categorical)*: Metamorphic facies
   - `weathering_class` *(Categorical)*: Weathering profile
   - `fold_position` *(Categorical)*: Structural limb / crest position
   - `structural_orientation` *(Categorical)*: Regional strike orientation

### Removed Shortcuts & Proxies
All ungrounded structural distance proxies (fault distance, shear-zone proximity, fold-hinge distance), synthetic directional aspect biases, and ASTER proxies were permanently purged. Sensitivity benchmarks demonstrated that retaining synthetic occurrence distance artificially inflated PR-AUC by +0.2345 through coordinate shortcutting.

---

## 3. Spatial Validation Strategy & Results

- **Spatial Tuning**: 5-Fold `GroupKFold` on `spatial_block_id` (zero leakage between blocks).
- **Evaluation Holdout**: 80/20 `GroupShuffleSplit` on `spatial_block_id`.

### Holdout Performance:
| Metric | Score | Note |
|:---|:---:|:---|
| **ROC-AUC** | **0.8917** | Robust discriminative capacity on unseen spatial blocks |
| **PR-AUC** | **0.3717** | ~5.5x over random baseline (prevalence = 6.7%) |
| **Balanced Accuracy** | **0.8058** | Balanced sensitivity across positive & background |
| **Recall** | **0.6975** | Captures ~70% of held-out manganese occurrences |
| **Precision** | **0.4338** | High precision in heavily imbalanced PU regime |
| **F1-Score** | **0.5349** | Harmonic balance between precision and recall |

### Empirical Probability Thresholds:
- **Moderate Prospectivity**: Model Score ≥ `0.1718` (80th percentile of test predictions)
- **High Prospectivity**: Model Score ≥ `0.9163` (95th percentile of test predictions)

---

## 4. Pipeline Execution Commands

```bash
# Train model and generate metrics.json + feature_importance.csv
python3 train_model.py

# Run comprehensive test suite (unit, ML, frontend, satellite)
pytest -v

# Run production validation & acceptance test suite
python3 evaluate_production_system.py

# Single-point inference via CLI
python3 predict.py --lat 21.8400 --lon 80.2311
```
