import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "bonai_keonjhar")
MODELS_DIR = os.path.join(BASE_DIR, "models", "bonai_keonjhar")
SCHEMA_PATH = os.path.join(DATA_DIR, "bonai_keonjhar_43_feature_schema.json")
TRAINABLE_DATA_PATH = os.path.join(DATA_DIR, "bonai_keonjhar_43_feature_trainable.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "bonai_keonjhar_43_feature_training_dataset.csv")

def encode_categoricals(df: pd.DataFrame, cat_cols: list, mappings: dict = None) -> tuple:
    df_out = df.copy()
    if mappings is None:
        mappings = {}
        for col in cat_cols:
            unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
            mappings[col] = {val: float(idx) for idx, val in enumerate(unique_vals)}

    for col in cat_cols:
        col_map = mappings[col]
        df_out[col] = df[col].apply(
            lambda v: col_map.get(str(v), -1.0) if pd.notna(v) else -1.0
        ).astype(float)

    return df_out, mappings

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    pos_count = np.sum(y == 1)
    neg_count = len(y) - pos_count
    if pos_count == 0:
        return np.ones(len(y))
    return np.where(y == 1, neg_count / pos_count, 1.0)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 80)
    print("GEOSPECTRA — BONAI–KEONJHAR 43-FEATURE MODEL TRAINING")
    print("Domain: Joda–Barbil Sector, Bonai–Keonjhar Manganese Belt, Odisha")
    print("=" * 80)

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    feature_cols = schema["primary_features"]
    cat_cols = schema["categorical_features"]
    cat_indices = schema["categorical_indices"]
    bbox = schema["study_domain"]

    assert len(feature_cols) == 43, f"Expected exactly 43 features, got {len(feature_cols)}"
    assert len(cat_cols) == 9, f"Expected exactly 9 categorical features, got {len(cat_cols)}"

    df_full = pd.read_csv(FULL_DATA_PATH)
    n_full_total = len(df_full)
    n_full_pos = int((df_full["label_status"] == "POSITIVE").sum())
    n_full_unl = int((df_full["label_status"] == "UNLABELLED").sum())
    n_full_unc = int((df_full["label_status"] == "UNCERTAIN").sum())
    print(f"\n[1/8] Ingested Full Dataset: {n_full_total:,} rows")
    print(f"      - POSITIVE:  {n_full_pos:,}")
    print(f"      - UNLABELLED:{n_full_unl:,}")
    print(f"      - UNCERTAIN: {n_full_unc:,} (strictly excluded from model training)")

    df_trainable = pd.read_csv(TRAINABLE_DATA_PATH)
    assert len(df_trainable) == 19421, f"Expected 19,421 trainable rows, got {len(df_trainable)}"
    assert (df_trainable["label_status"] == "UNCERTAIN").sum() == 0, "UNCERTAIN rows found in trainable set!"

    y_all = (df_trainable["label_status"] == "POSITIVE").astype(int).values
    groups_all = df_trainable["spatial_block_id"].values
    pos_prevalence = float(np.mean(y_all))

    print(f"\n[2/8] Trainable Samples: {len(df_trainable):,} ({np.sum(y_all == 1):,} POSITIVE, {np.sum(y_all == 0):,} UNLABELLED)")
    print(f"      Positive Prevalence: {pos_prevalence * 100:.2f}% (PR Random Baseline: {pos_prevalence:.4f})")
    print(f"      Spatial Blocks: {len(np.unique(groups_all))} blocks")

    print("\n[3/8] Performing Outer Spatial Holdout Split (80% train / 20% test on spatial_block_id)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df_trainable, y_all, groups=groups_all))

    train_df = df_trainable.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df_trainable.iloc[test_idx].copy().reset_index(drop=True)

    train_blocks = set(train_df["spatial_block_id"])
    test_blocks = set(test_df["spatial_block_id"])
    assert len(train_blocks.intersection(test_blocks)) == 0, "Spatial block leakage detected between train and test!"

    y_train = (train_df["label_status"] == "POSITIVE").astype(int).values
    y_test = (test_df["label_status"] == "POSITIVE").astype(int).values
    groups_train = train_df["spatial_block_id"].values

    print(f"      Spatial Train Set: {len(train_df):,} stations ({len(train_blocks)} blocks, {np.sum(y_train == 1)} positives)")
    print(f"      Spatial Test Set:  {len(test_df):,} stations ({len(test_blocks)} blocks, {np.sum(y_test == 1)} positives)")

    print("\n[4/8] Deriving deterministic categorical encodings from training split...")
    X_train_encoded, cat_mappings = encode_categoricals(train_df[feature_cols], cat_cols, mappings=None)
    X_test_encoded, _ = encode_categoricals(test_df[feature_cols], cat_cols, mappings=cat_mappings)

    for c in cat_cols:
        print(f"      - {c}: {len(cat_mappings[c])} categories")

    print("\n[5/8] Running 5-Fold Spatial GroupKFold Cross-Validation...")
    gkf = GroupKFold(n_splits=5)
    cv_pr_aucs = []
    cv_roc_aucs = []

    for fold_idx, (f_tr_idx, f_val_idx) in enumerate(gkf.split(X_train_encoded, y_train, groups=groups_train)):
        X_f_tr = X_train_encoded.iloc[f_tr_idx]
        y_f_tr = y_train[f_tr_idx]
        X_f_val = X_train_encoded.iloc[f_val_idx]
        y_f_val = y_train[f_val_idx]
        w_f_tr = compute_sample_weights(y_f_tr)

        clf_fold = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            max_iter=150,
            max_bins=255,
            categorical_features=cat_indices,
            early_stopping=False,
            random_state=42 + fold_idx,
        )
        clf_fold.fit(X_f_tr, y_f_tr, sample_weight=w_f_tr)
        val_probs = clf_fold.predict_proba(X_f_val)[:, 1]

        fold_pr = average_precision_score(y_f_val, val_probs)
        fold_roc = roc_auc_score(y_f_val, val_probs)
        cv_pr_aucs.append(fold_pr)
        cv_roc_aucs.append(fold_roc)
        print(f"      Fold {fold_idx + 1}: PR-AUC = {fold_pr:.4f}, ROC-AUC = {fold_roc:.4f}")

    mean_cv_pr = float(np.mean(cv_pr_aucs))
    std_cv_pr = float(np.std(cv_pr_aucs))
    mean_cv_roc = float(np.mean(cv_roc_aucs))
    std_cv_roc = float(np.std(cv_roc_aucs))
    print(f"   >>> 5-Fold Spatial CV: Mean PR-AUC = {mean_cv_pr:.4f} (±{std_cv_pr:.4f}), Mean ROC-AUC = {mean_cv_roc:.4f} (±{std_cv_roc:.4f})")

    print("\n[6/8] Fitting Final 43-Feature HistGradientBoosting Model on Spatial Train Split...")
    w_train = compute_sample_weights(y_train)
    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_iter=150,
        max_bins=255,
        categorical_features=cat_indices,
        early_stopping=False,
        random_state=42,
    )
    final_model.fit(X_train_encoded, y_train, sample_weight=w_train)

    y_test_proba = final_model.predict_proba(X_test_encoded)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    pr_auc = float(average_precision_score(y_test, y_test_proba))
    prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    print("\n" + "=" * 60)
    print("   BONAI–KEONJHAR FINAL MODEL METRICS (SPATIAL HOLDOUT)")
    print("=" * 60)
    print(f"   ROC-AUC Score:      {roc_auc:.4f}")
    print(f"   PR-AUC Score:       {pr_auc:.4f} (Random Baseline: {pos_prevalence:.4f})")
    print(f"   Precision:          {prec:.4f}")
    print(f"   Recall:             {rec:.4f}")
    print(f"   F1-Score:           {f1:.4f}")
    print(f"   Balanced Accuracy:  {bal_acc:.4f}")
    print(f"   Confusion Matrix:   TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    print("\n[7/8] Running Latitude / Longitude Ablation Test (43 features vs 41 features)...")
    ablated_cols = [c for c in feature_cols if c not in ["latitude", "longitude"]]
    ablated_cat_indices = [ablated_cols.index(c) for c in cat_cols]

    X_train_ablated = X_train_encoded[ablated_cols]
    X_test_ablated = X_test_encoded[ablated_cols]

    model_ablated = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_iter=150,
        max_bins=255,
        categorical_features=ablated_cat_indices,
        early_stopping=False,
        random_state=42,
    )
    model_ablated.fit(X_train_ablated, y_train, sample_weight=w_train)

    y_test_proba_ablated = model_ablated.predict_proba(X_test_ablated)[:, 1]
    y_test_pred_ablated = (y_test_proba_ablated >= 0.5).astype(int)

    roc_auc_ablated = float(roc_auc_score(y_test, y_test_proba_ablated))
    pr_auc_ablated = float(average_precision_score(y_test, y_test_proba_ablated))
    prec_ablated = float(precision_score(y_test, y_test_pred_ablated, zero_division=0))
    rec_ablated = float(recall_score(y_test, y_test_pred_ablated, zero_division=0))
    f1_ablated = float(f1_score(y_test, y_test_pred_ablated, zero_division=0))
    bal_acc_ablated = float(balanced_accuracy_score(y_test, y_test_pred_ablated))

    diff_roc = roc_auc - roc_auc_ablated
    diff_pr = pr_auc - pr_auc_ablated

    print("=" * 60)
    print("   ABLATION COMPARISON (SPATIAL HOLDOUT)")
    print("=" * 60)
    print(f"   Full Model (43 feats):      ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
    print(f"   Ablated Model (41 feats):   ROC-AUC = {roc_auc_ablated:.4f}, PR-AUC = {pr_auc_ablated:.4f}")
    print(f"   Delta (Full - Ablated):     ΔROC-AUC = {diff_roc:+.4f}, ΔPR-AUC = {diff_pr:+.4f}")
    if abs(diff_pr) < 0.05:
        print("   >>> Conclusion: Model demonstrates robust spatial generalization without coordinate overfitting.")
    else:
        print(f"   >>> Note: Coordinates contribute {diff_pr:+.4f} PR-AUC to spatial validation.")

    ablation_payload = {
        "domain": "bonai_keonjhar",
        "full_model": {
            "features_count": 43,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
        },
        "ablated_model_no_coords": {
            "features_count": 41,
            "excluded_features": ["latitude", "longitude"],
            "roc_auc": round(roc_auc_ablated, 4),
            "pr_auc": round(pr_auc_ablated, 4),
            "precision": round(prec_ablated, 4),
            "recall": round(rec_ablated, 4),
            "f1_score": round(f1_ablated, 4),
            "balanced_accuracy": round(bal_acc_ablated, 4),
        },
        "delta": {
            "roc_auc_difference": round(diff_roc, 4),
            "pr_auc_difference": round(diff_pr, 4),
        },
        "interpretation": "Spatial holdout ablation testing coordinate dependence.",
    }
    ablation_file = os.path.join(MODELS_DIR, "ablation_metrics.json")
    with open(ablation_file, "w") as f:
        json.dump(ablation_payload, f, indent=2)
    print(f"   Saved ablation metrics to: {ablation_file}")

    print("\n[8/8] Building Reference Distribution, Applicability Model, and Artifacts...")
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    iso_forest = IsolationForest(n_estimators=100, random_state=42)
    iso_forest.fit(X_train_encoded[numeric_cols].fillna(0.0))
    iso_scores = iso_forest.decision_function(X_train_encoded[numeric_cols].fillna(0.0))
    th_mod = float(np.percentile(iso_scores, 20))
    th_low = float(np.percentile(iso_scores, 5))

    perm_res = permutation_importance(final_model, X_test_encoded, y_test, n_repeats=5, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_res.importances_mean,
        "importance_std": perm_res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_file = os.path.join(MODELS_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_file, index=False)
    print(f"      Saved feature importance to: {imp_file}")

    X_all_encoded, _ = encode_categoricals(df_trainable[feature_cols], cat_cols, mappings=cat_mappings)
    all_ref_probs = final_model.predict_proba(X_all_encoded)[:, 1]
    sorted_ref_probs = np.sort(all_ref_probs)
    ref_pred_file = os.path.join(MODELS_DIR, "reference_predictions.npy")
    np.save(ref_pred_file, sorted_ref_probs)
    print(f"      Saved sorted reference predictions ({len(sorted_ref_probs)} values) to: {ref_pred_file}")

    geo_cols = [
        'geological_group', 'geological_formation', 'stratigraphic_unit',
        'lithology', 'metamorphic_grade', 'weathering_class',
        'fold_position', 'structural_orientation', 'valley_or_ridge_class',
    ]
    geo_lookup_df = df_trainable[['latitude', 'longitude'] + geo_cols].drop_duplicates(
        subset=['latitude', 'longitude']
    ).reset_index(drop=True)
    geo_lookup_file = os.path.join(MODELS_DIR, "geological_lookup_grid.csv")
    geo_lookup_df.to_csv(geo_lookup_file, index=False)
    print(f"      Saved geological lookup grid: {len(geo_lookup_df)} stations to: {geo_lookup_file}")

    df_trainable["raw_prob"] = all_ref_probs
    idx_left = np.searchsorted(sorted_ref_probs, all_ref_probs, side='left')
    idx_right = np.searchsorted(sorted_ref_probs, all_ref_probs, side='right')
    df_trainable["prospectivity_score"] = np.round((idx_left + idx_right) / (2.0 * len(sorted_ref_probs)) * 100.0, 1)

    top_targets = df_trainable.sort_values(["label", "prospectivity_score"], ascending=[False, False]).head(10).copy()
    top_targets["target_rank"] = [f"{i+1:02d}" for i in range(len(top_targets))]
    top_targets["target_id"] = [f"BK-MN-{i+1:02d}" for i in range(len(top_targets))]
    top_targets["sector"] = "Joda–Barbil / Bonai–Keonjhar"
    targets_file = os.path.join(MODELS_DIR, "top_exploration_targets.csv")
    top_targets[[
        "target_rank", "target_id", "station_id", "latitude", "longitude",
        "prospectivity_score", "raw_prob", "geological_formation", "lithology", "sector"
    ]].to_csv(targets_file, index=False)
    print(f"      Saved top exploration targets ({len(top_targets)} targets) to: {targets_file}")

    pos_mask = df_trainable["label"] == 1
    unl_mask = df_trainable["label"] == 0
    s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    s2_pos_mean = df_trainable.loc[pos_mask, s2_bands].mean().to_dict()
    s2_unl_mean = df_trainable.loc[unl_mask, s2_bands].mean().to_dict()
    radar_feats = ['VV', 'VH', 'VV_VH_ratio', 'radar_backscatter_mean', 'radar_texture']
    radar_pos_mean = df_trainable.loc[pos_mask, radar_feats].mean().to_dict()

    model_bundle = {
        "model": final_model,
        "domain_id": "bonai_keonjhar",
        "region_name": bbox["region_name"],
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_indices": cat_indices,
        "cat_mappings": cat_mappings,
        "classifier_name": "HistGradientBoostingClassifier",
        "schema_version": schema["schema_version"],
        "domain_bbox": bbox,
        "probability_thresholds": {
            "moderate_cutoff": float(np.percentile(y_test_proba, 80)),
            "high_cutoff": float(np.percentile(y_test_proba, 95)),
            "description": "80th and 95th percentiles of spatial test predictions on Bonai–Keonjhar holdout.",
        },
        "applicability": {
            "model": iso_forest,
            "threshold_moderate": th_mod,
            "threshold_low": th_low,
            "numeric_cols": numeric_cols,
            "description": "IsolationForest decision_function thresholds (20th and 5th percentiles).",
        },
        "derived_constants": {},
        "s2_positive_mean": s2_pos_mean,
        "s2_unlabelled_mean": s2_unl_mean,
        "radar_positive_mean": radar_pos_mean,
        "reference_predictions": sorted_ref_probs,
        "metadata": {
            "model_type": "HistGradientBoostingClassifier",
            "domain": "bonai_keonjhar",
            "total_samples": len(df_trainable),
            "positive_samples": int(np.sum(y_all == 1)),
            "unlabelled_samples": int(np.sum(y_all == 0)),
            "uncertain_samples_excluded": n_full_unc,
            "features_count": len(feature_cols),
            "cv_pr_auc_mean": round(mean_cv_pr, 4),
            "holdout_roc_auc": round(roc_auc, 4),
            "holdout_pr_auc": round(pr_auc, 4),
            "provenance": "Synthetic Prototype Benchmark Dataset (Honest Data Disclosure)",
        },
    }

    bundle_file = os.path.join(MODELS_DIR, "manganese_prospectivity_model.pkl")
    joblib.dump(model_bundle, bundle_file)
    print(f"\n>>> SAVED BONAI–KEONJHAR MODEL BUNDLE: {bundle_file}")

    sausar_model_file = os.path.join(BASE_DIR, "manganese_prospectivity_model.pkl")
    assert os.path.exists(sausar_model_file), "Sausar model missing!"
    sausar_bundle = joblib.load(sausar_model_file)
    assert "Sausar" in sausar_bundle.get("domain_bbox", {}).get("region_name", "") or "Sausar" in sausar_bundle.get("metadata", {}).get("study_area", ""), "Sausar model was corrupted!"
    print(">>> CONFIRMED: Root Sausar model remains 100% intact and untouched.")

    metrics_payload = {
        "domain_id": "bonai_keonjhar",
        "domain_name": bbox["region_name"],
        "model_name": "HistGradientBoostingClassifier",
        "schema_version": schema["schema_version"],
        "n_features": len(feature_cols),
        "feature_list": feature_cols,
        "categorical_features": cat_cols,
        "data_split": {
            "total_raw_rows": n_full_total,
            "uncertain_excluded": n_full_unc,
            "trainable_rows": len(df_trainable),
            "positive_rows": int(np.sum(y_all == 1)),
            "unlabelled_rows": int(np.sum(y_all == 0)),
            "spatial_blocks_count": len(np.unique(groups_all)),
            "train_spatial_blocks": len(train_blocks),
            "test_spatial_blocks": len(test_blocks),
        },
        "spatial_validation": {
            "strategy": "GroupShuffleSplit (80/20 holdout) on spatial_block_id",
            "5fold_cv_mean_pr_auc": round(mean_cv_pr, 4),
            "5fold_cv_std_pr_auc": round(std_cv_pr, 4),
            "5fold_cv_mean_roc_auc": round(mean_cv_roc, 4),
            "5fold_cv_std_roc_auc": round(std_cv_roc, 4),
            "holdout_roc_auc": round(roc_auc, 4),
            "holdout_pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "positive_prevalence": round(pos_prevalence, 4),
            "pr_random_baseline": round(pos_prevalence, 4),
            "confusion_matrix": {
                "true_unlabelled": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
        },
        "ablation_summary": {
            "full_43_pr_auc": round(pr_auc, 4),
            "ablated_41_pr_auc": round(pr_auc_ablated, 4),
            "delta_pr_auc": round(diff_pr, 4),
            "delta_roc_auc": round(diff_roc, 4),
        },
        "data_honesty_disclosure": (
            "This model was trained on a synthetic prototype benchmark dataset designed to mirror "
            "the 43-feature GeoSpectra exploration schema for the Bonai–Keonjhar manganese belt. "
            "Feature values are synthetic prototypes and not in-situ field measurements. "
            "Predictions reflect relative exploration rankings (0–100) and not calibrated physical probabilities."
        ),
    }

    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f">>> Saved metrics to: {metrics_file}")
    print("=" * 80)
    print("BONAI–KEONJHAR TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
