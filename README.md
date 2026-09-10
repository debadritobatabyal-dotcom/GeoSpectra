# GeoSpectra | Manganese Exploration Intelligence System
### AI/ML & Satellite Remote Sensing for Prospectivity Mapping in the Central Indian Sausar Manganese Belt

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/frontend-Streamlit%201.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn%201.7+-F7931E.svg)](https://scikit-learn.org/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-Sentinel--1%20%7C%20Sentinel--2%20%7C%20SRTM-34A853.svg)](https://earthengine.google.com/)
[![Tests](https://img.shields.io/badge/tests-30%20passed%20%E2%9C%93-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary

Manganese is a critical mineral vital to global steel manufacturing, high-energy-density EV batteries, and national economic security. In India, domestic manganese demand is high while approximately 50% of high-grade ore requirements depend on international imports.

**GeoSpectra** is an authoritative, end-to-end mineral prospectivity exploration system engineered for **MOIL Limited** and the **Central Indian Sausar Manganese Belt** (Balaghat–Tirodi–Bhandara region: 20.95°N–22.15°N, 79.35°E–80.65°E).

The platform transforms raw spaceborne observations into actionable exploration targets by fusing:
1. **Sentinel-2 MSI Level-2A** multispectral reflectance and diagnostic alteration band ratios (SWIR/VNIR/Red-Edge).
2. **Sentinel-1 C-Band SAR** dual-polarization radar backscatter and surface texture (VV, VH).
3. **SRTM 30m Digital Elevation Model (DEM)** geomorphometry and Topographic Position Index (TPI).
4. **Geological Survey of India (GSI)** published lithological and stratigraphic crosswalk mapping.

```
Coordinate Selection (Lat, Lon)
            ↓
   Domain Boundary Check (20.95–22.15°N, 79.35–80.65°E)
            ↓
  Feature Extraction (Sentinel-1 SAR + Sentinel-2 MSI + SRTM DEM + GSI Geology)
            ↓
  Authoritative 43-Feature Vector Assembly (Schema v4.0.0)
            ↓
  Deterministic Categorical Encoding (Saved Integer Mappings)
            ↓
  HistGradientBoostingClassifier (Spatially Validated PU Model)
            ↓
  Relative Prospectivity Score (0–100 Percentile Rank)
            ↓
  Multi-Criteria Decision Intelligence:
  ├── Prospectivity Classification (Very Low, Low, Moderate, High, Very High)
  ├── IsolationForest Environmental Applicability (High / Moderate / Low)
  ├── TreeExplainer SHAP Top-5 Feature Attribution (Non-Causal Insights)
  └── Nearest Documented MOIL Mine Context (For Literature Reference)
```

---

## 2. Key Highlights & Scientific Grounding

- **Zero Synthetic Shortcuts or Leakage**: All ungrounded structural distance proxies (fault distance, shear-zone proximity, fold-hinge distance) and synthetic directional aspect biases have been permanently purged. A rigorous sensitivity audit confirmed that synthetic occurrence distance artificially inflated PR-AUC by +0.2345 through coordinate memorization.
- **Aspect Invariance Guaranteed**: Slope aspect direction has mathematically zero impact on prospectivity scores (`Δ = 0.000000`), ensuring that south-facing terrain is never unfairly penalized.
- **Strict Spatial Cross-Validation**: Validated using 5-Fold `GroupKFold` and 80/20 `GroupShuffleSplit` on `spatial_block_id` to guarantee zero spatial autocorrelation leakage between training and evaluation blocks.
- **Separation Between Score and Applicability**: Features out-of-distribution (OOD) are evaluated independently via an `IsolationForest` applicability index, alerting geologists when terrain departs from the training distribution.
- **Fail-Safe Satellite Guardrail**: If live satellite APIs are unreachable, the system explicitly reports `SATELLITE_UNAVAILABLE` rather than fabricating synthetic observations.

---

## 3. Validated Model Performance

Evaluation on an 80/20 spatial holdout (`spatial_block_id`) demonstrates strong discriminative performance in an imbalanced Positive-Unlabeled (PU) exploration regime:

| Metric | Holdout Score | Benchmark Significance |
|:---|:---:|:---|
| **ROC-AUC** | **0.8917** | Outstanding discriminative ability on unseen spatial blocks |
| **PR-AUC** | **0.3717** | **~5.5x gain** over random baseline (positive prevalence: 6.7%) |
| **Balanced Accuracy** | **0.8058** | Balanced sensitivity across positive horizons and background |
| **Recall** | **0.6975** | Captures ~70% of documented manganese occurrences |
| **Precision** | **0.4338** | High precision given strict Positive-Unlabeled constraints |
| **F1-Score** | **0.5349** | Strong harmonic balance between precision and recall |

### Empirical Probability Cutoffs:
- **Moderate Prospectivity**: Score ≥ **0.1718** (80th percentile of test predictions)
- **High Prospectivity**: Score ≥ **0.9163** (95th percentile of test predictions)

### Known MOIL Mine Verification:
In systematic production evaluation, confirmed MOIL mines (Bharveli, Balaghat, Ukwa, Tirodi, Dongri Buzurg) achieve a **median prospectivity score of 97.4/100**, achieving a **+45.3 point separation** over regional background stations (median: 52.1/100).

---

## 4. Repository Structure

```
ml_mn_full_project/
├── app.py                          # Full-featured Streamlit exploration dashboard
├── predict.py                      # Authoritative single-location inference engine
├── feature_pipeline.py             # Feature extraction (GEE live + SQLite cache + GSI geology)
├── train_model.py                  # Spatial-block PU training pipeline & cross-validation
├── evaluate_production_system.py   # Comprehensive acceptance test suite (12/12 verified)
├── authenticate_gee.py             # Interactive Google Earth Engine authentication helper
├── round_trip_test.py              # End-to-end round-trip verification script
│
├── feature_schema.json             # Authoritative 43-feature schema specification (v4.0.0)
├── geology_crosswalk.json          # GSI stratigraphy & formation classification crosswalk
├── geological_lookup_grid.csv      # Spatial nearest-neighbor geological grid lookup (GSI)
├── known_occurrences.csv           # Documented MOIL mines (for literature display only)
├── manganese_prospectivity_model.pkl # Production model bundle (HistGBM + IsolationForest)
├── feature_importance.csv          # Permutation feature importances across domains
├── metrics.json                    # Spatial holdout performance & winning hyperparameters
│
├── data/
│   ├── dataset/
│   │   ├── sausar_manganese_core_features.csv    # 20,000-station curated training dataset
│   │   ├── sausar_manganese_extended_features.csv # Extended geochemical features
│   │   └── sausar_manganese_data_dictionary.csv  # Feature definitions and units
│   ├── manganese_prospectivity_raster.png       # Continuous regional prospectivity heat surface
│   ├── manganese_prospectivity_grid.csv         # Systematic exploration grid
│   └── Manganese_AI_Space_Tech_Research_Report.md # Full scientific research report
│
├── tests/
│   ├── conftest.py                 # Pytest environment & module configuration
│   ├── test_data.py                # Dataset integrity & banned feature exclusion tests
│   ├── test_feature_pipeline.py    # Categorical encoding & schema alignment tests
│   ├── test_frontend.py            # Streamlit AppTest interface & auth validation
│   ├── test_geo.py                 # Domain boundary guardrails & spatial lookup tests
│   ├── test_ml.py                  # Model serialization, PU separation & aspect invariance
│   └── test_satellite.py          # Satellite cache, GEE error handling & guardrail tests
│
├── .env.example                    # Template environment variables for Earth Engine
├── .gitignore                      # Comprehensive Git ignore rules
├── requirements.txt                # Categorized Python dependencies
├── LICENSE                         # MIT License
├── README.md                       # Main project documentation
└── README_ML.md                    # Machine learning pipeline technical details
```

---

## 5. Quickstart & Installation

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/debadritobatabyal-dotcom/GeoSpectra.git
cd GeoSpectra

python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Google Earth Engine (Optional for Live Imagery)
If you wish to query live Sentinel-1, Sentinel-2, and SRTM observations via Google Earth Engine:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set your Google Cloud Project ID:
   ```env
   EARTHENGINE_PROJECT=your-google-cloud-project-id
   ```
3. Run the interactive Earth Engine authenticator:
   ```bash
   python3 authenticate_gee.py
   ```

*(Note: The system functions completely out-of-the-box in offline mode using the pre-extracted curated dataset, spatial geological grid lookup, and cached feature database).*

---

## 6. Usage & Execution

### A. Launch Interactive Web Portal
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- Click **"Demo Access"** to immediately enter the dashboard.
- Interactive Folium map with satellite basemap, MOIL mine overlays, and prospectivity surface.
- Click any coordinate on the map or enter latitude/longitude to generate predictions.
- View interactive SHAP feature attributions, mineral ratios, radar backscatter, and geological context.

### B. Single-Point Inference via CLI
```bash
python3 predict.py --lat 21.8400 --lon 80.2311
```
Output:
```json
{
  "status": "SUCCESS",
  "latitude": 21.84,
  "longitude": 80.2311,
  "prospectivity_score": 97.9,
  "prospectivity_class": "VERY HIGH",
  "applicability_status": "HIGH APPLICABILITY",
  "applicability_score": 0.0528,
  "key_drivers": [
    "Feature 'latitude' increased prospectivity score (impact: +1.284)",
    "Feature 'geological_group' increased prospectivity score (impact: +0.642)",
    "Feature 'BSI' increased prospectivity score (impact: +0.218)"
  ]
}
```

### C. Run Test Suite
```bash
pytest -v
```
Runs 30 comprehensive unit and integration tests across data integrity, machine learning, frontend UI, geospatial boundaries, and satellite guardrails:
```
======================== 30 passed in 2.15s ========================
```

### D. Run Acceptance Test Suite
```bash
python3 evaluate_production_system.py
```
Evaluates 1,000 spatial points, compares live vs training feature distributions, audits all known MOIL mines, identifies greenfield target corridors, and verifies all 12 acceptance criteria:
```
================================================================================
ALL 12 ACCEPTANCE TESTS PASSED (12/12) ✓✓✓
================================================================================
```

---

## 7. Technology Stack

- **Machine Learning**: `scikit-learn` (`HistGradientBoostingClassifier`, `IsolationForest`), `shap` (`TreeExplainer`), `scipy`.
- **Remote Sensing & GIS**: `earthengine-api` (Sentinel-1 SAR, Sentinel-2 MSI Level-2A, SRTM 30m DEM), `folium`, `streamlit-folium`.
- **Web Application & UI**: `streamlit`, `plotly`, Vanilla CSS with glassmorphic cards and typography.
- **Data Engineering & Caching**: `pandas`, `numpy`, `sqlite3` (TTL feature cache), `joblib`.
- **Testing & Verification**: `pytest`, `streamlit.testing.v1.AppTest`.

---

## 8. Exploration & Ethical Disclosures

1. **Screening Tool Only**: The prospectivity scores generated by this system represent **relative exploration ranking probabilities** (0–100 percentile rank) across the Sausar belt. They do **not** constitute economic reserve estimations or subsurface mineral grade certifications.
2. **Subsurface Confirmation**: Satellite remote sensing and surface geomorphometry detect surface alterations, regolith signatures, and stratigraphic host settings. Ground-truthing through core drilling, geophysical surveys, and chemical assays is essential prior to investment decisions.
3. **Environmental & Protected Areas**: Field teams must cross-reference exploration target corridors with local forest department boundaries and wildlife sanctuaries before conducting ground reconnaissance.

---

## 9. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
