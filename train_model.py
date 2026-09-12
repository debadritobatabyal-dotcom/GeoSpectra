import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    IsolationForest,
)
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

def get_feature_domain(feature_name: str) -> str:
    if feature_name in ['latitude', 'longitude']:
        return 'Spatial Coordinates'
    elif feature_name in [
        'elevation_m', 'slope_deg', 'curvature', 'terrain_ruggedness',
        'local_relief_m', 'topographic_position_index', 'valley_or_ridge_class',
    ]:
        return 'Terrain & DEM'
    elif feature_name in [
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
        'NDVI', 'NDMI', 'MNDWI', 'BSI', 'B4_B2_ratio', 'B11_B12_ratio',
        'B12_B8_ratio', 'B11_B8_ratio', 'gossan_alteration_index',
        'red_edge_ratio_1', 'red_edge_ratio_2',
    ]:
        return 'Sentinel-2 Optical & Indices'
    elif feature_name in [
        'VV', 'VH', 'VV_VH_ratio', 'radar_backscatter_mean', 'radar_texture',
    ]:
        return 'Sentinel-1 SAR Radar'
    elif feature_name in [
        'geological_group', 'geological_formation', 'stratigraphic_unit',
        'lithology', 'metamorphic_grade', 'weathering_class',
        'fold_position', 'structural_orientation',
    ]:
        return 'Geological Map (GSI)'
    else:
        return 'Other'

def encode_categoricals(df: pd.DataFrame, cat_cols: list, mappings: dict = None) -> tuple:
    df_out = df.copy()
    if mappings is None:
        mappings = {}
        for col in cat_cols:
            unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
            mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

    for col in cat_cols:
        col_map = mappings[col]
        df_out[col] = df[col].apply(
            lambda v: col_map.get(str(v), -1) if pd.notna(v) else -1
        ).astype(float)

    return df_out, mappings

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    pos_count = np.sum(y == 1)
    neg_count = len(y) - pos_count
    return np.where(y == 1, neg_count / pos_count, 1.0)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "feature_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    feature_cols = schema["primary_features"]
    cat_cols = schema["categorical_features"]
    cat_indices = schema["categorical_indices"]

    print("=" * 80)
    print("MOIL MANGANESE PROSPECTIVITY PREDICTION — MODEL TRAINING")
    print(f"Model Architecture: Multi-Classifier Comparison ({len(feature_cols)} features)")
    print(f"Schema version: {schema['schema_version']}")
    print("=" * 80)

    data_path = os.path.join(base_dir, "data", "dataset", "sausar_manganese_core_features.csv")
    raw_df = pd.read_csv(data_path)
    print(f"[1/9] Loaded {len(raw_df):,} total raw records from {data_path}")

    clean_df = raw_df[raw_df["label_status"].isin(["POSITIVE", "UNLABELLED"])].copy()
    clean_df["target"] = (clean_df["label_status"] == "POSITIVE").astype(int)
    print(f"      Filtered PU dataset: {len(clean_df):,} samples "
          f"(POSITIVE: {clean_df['target'].sum():,}, UNLABELLED: {(clean_df['target'] == 0).sum():,})")

    print("\n[2/9] Creating outer spatial holdout split (80% train / 20% test on spatial_block_id)...")
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(
        gss_outer.split(clean_df, clean_df["target"], groups=clean_df["spatial_block_id"])
    )

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)

    train_blocks = set(train_df["spatial_block_id"])
    test_blocks = set(test_df["spatial_block_id"])
    assert len(train_blocks.intersection(test_blocks)) == 0, "Spatial block leakage detected!"

    print(f"      Train set: {len(train_df):,} samples ({len(train_blocks)} spatial blocks)")
    print(f"      Test set:  {len(test_df):,} samples ({len(test_blocks)} spatial blocks)")

    print("\n[3/9] Deriving deterministic categorical integer mappings from training set...")
    X_train_raw = train_df[feature_cols]
    y_train = train_df["target"].values
    X_test_raw = test_df[feature_cols]
    y_test = test_df["target"].values

    X_train_encoded, cat_mappings = encode_categoricals(X_train_raw, cat_cols, mappings=None)
    X_test_encoded, _ = encode_categoricals(X_test_raw, cat_cols, mappings=cat_mappings)

    for col in cat_cols:
        n_cats = len(cat_mappings[col])
        print(f"      {col}: {n_cats} categories")

    print("\n[4/9] Multi-classifier comparison using 5-fold Spatial GroupKFold...")
    gkf = GroupKFold(n_splits=5)
    groups_train = train_df["spatial_block_id"].values
    w_train = compute_sample_weights(y_train)

    classifiers = {
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_iter=150,
            max_bins=255,
            categorical_features=cat_indices,
            early_stopping=False,
            random_state=42,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_results = {}
    for clf_name, clf_factory in classifiers.items():
        fold_pr_aucs = []
        fold_roc_aucs = []
        for fold_train_idx, fold_val_idx in gkf.split(X_train_encoded, y_train, groups=groups_train):
            X_f_tr = X_train_encoded.iloc[fold_train_idx]
            y_f_tr = y_train[fold_train_idx]
            X_f_val = X_train_encoded.iloc[fold_val_idx]
            y_f_val = y_train[fold_val_idx]

            clf = clf_factory()
            if clf_name == "HistGradientBoosting":
                weights_f = compute_sample_weights(y_f_tr)
                clf.fit(X_f_tr, y_f_tr, sample_weight=weights_f)
            else:
                clf.fit(X_f_tr, y_f_tr)

            val_preds = clf.predict_proba(X_f_val)[:, 1]
            fold_pr_aucs.append(average_precision_score(y_f_val, val_preds))
            fold_roc_aucs.append(roc_auc_score(y_f_val, val_preds))

        mean_pr = float(np.mean(fold_pr_aucs))
        std_pr = float(np.std(fold_pr_aucs))
        mean_roc = float(np.mean(fold_roc_aucs))
        comparison_results[clf_name] = {
            "mean_pr_auc": mean_pr,
            "std_pr_auc": std_pr,
            "mean_roc_auc": mean_roc,
        }
        print(f"      {clf_name}: CV PR-AUC={mean_pr:.4f} (±{std_pr:.4f}), ROC-AUC={mean_roc:.4f}")

    best_clf_name = max(comparison_results, key=lambda k: comparison_results[k]["mean_pr_auc"])
    print(f"\n   >>> Selected classifier: {best_clf_name} (PR-AUC={comparison_results[best_clf_name]['mean_pr_auc']:.4f})")

    print(f"\n[5/9] Hyperparameter tuning for {best_clf_name}...")
    if best_clf_name == "HistGradientBoosting":
        param_grid = []
        for lr in [0.03, 0.06, 0.1]:
            for max_leaf in [15, 31, 63]:
                for min_leaf in [20, 50]:
                    for l2 in [0.0, 1.0]:
                        param_grid.append({
                            "learning_rate": lr,
                            "max_leaf_nodes": max_leaf,
                            "min_samples_leaf": min_leaf,
                            "l2_regularization": l2,
                        })

        grid_results = []
        for idx, params in enumerate(param_grid):
            fold_pr_aucs = []
            for f_train_idx, f_val_idx in gkf.split(X_train_encoded, y_train, groups=groups_train):
                X_f_tr = X_train_encoded.iloc[f_train_idx]
                y_f_tr = y_train[f_train_idx]
                X_f_val = X_train_encoded.iloc[f_val_idx]
                y_f_val = y_train[f_val_idx]
                weights_f = compute_sample_weights(y_f_tr)

                clf = HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=params["learning_rate"],
                    max_leaf_nodes=params["max_leaf_nodes"],
                    min_samples_leaf=params["min_samples_leaf"],
                    l2_regularization=params["l2_regularization"],
                    max_iter=150,
                    max_bins=255,
                    categorical_features=cat_indices,
                    early_stopping=False,
                    random_state=42,
                )
                clf.fit(X_f_tr, y_f_tr, sample_weight=weights_f)
                val_preds = clf.predict_proba(X_f_val)[:, 1]
                fold_pr_aucs.append(average_precision_score(y_f_val, val_preds))

            mean_pr = float(np.mean(fold_pr_aucs))
            std_pr = float(np.std(fold_pr_aucs))
            grid_results.append({"params": params, "mean_pr_auc": mean_pr, "std_pr_auc": std_pr})
            if (idx + 1) % 8 == 0 or (idx + 1) == len(param_grid):
                print(f"      Completed {idx + 1}/{len(param_grid)} parameter sets...")

        grid_results.sort(key=lambda r: (-r["mean_pr_auc"], r["std_pr_auc"]))
        best_res = grid_results[0]
        best_params = best_res["params"]
    else:

        best_params = {}
        best_res = {"mean_pr_auc": comparison_results[best_clf_name]["mean_pr_auc"], "std_pr_auc": comparison_results[best_clf_name].get("std_pr_auc", 0)}

    print(f"\n   Best hyperparameters: {best_params}")
    print(f"   Best CV PR-AUC: {best_res['mean_pr_auc']:.4f}")

    if best_clf_name == "HistGradientBoosting":
        print("\n[6/9] Manual early stopping on internal spatial validation split...")
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        inner_tr_idx, inner_val_idx = next(
            gss_inner.split(train_df, train_df["target"], groups=train_df["spatial_block_id"])
        )

        X_in_tr = X_train_encoded.iloc[inner_tr_idx]
        y_in_tr = y_train[inner_tr_idx]
        X_in_val = X_train_encoded.iloc[inner_val_idx]
        y_in_val = y_train[inner_val_idx]
        w_in_tr = compute_sample_weights(y_in_tr)

        warm_clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=best_params["learning_rate"],
            max_leaf_nodes=best_params["max_leaf_nodes"],
            min_samples_leaf=best_params["min_samples_leaf"],
            l2_regularization=best_params["l2_regularization"],
            max_bins=255,
            categorical_features=cat_indices,
            warm_start=True,
            max_iter=10,
            early_stopping=False,
            random_state=42,
        )

        best_val_pr = -1.0
        frozen_max_iter = 10
        no_improve_count = 0

        for iters in range(10, 501, 10):
            warm_clf.max_iter = iters
            warm_clf.fit(X_in_tr, y_in_tr, sample_weight=w_in_tr)
            val_proba = warm_clf.predict_proba(X_in_val)[:, 1]
            val_pr = average_precision_score(y_in_val, val_proba)

            if val_pr >= best_val_pr + 0.002:
                best_val_pr = val_pr
                frozen_max_iter = iters
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= 5:
                print(f"      Early stopping at max_iter={iters} (best: {frozen_max_iter}, PR-AUC {best_val_pr:.4f})")
                break

        print(f"      Frozen optimal iterations: {frozen_max_iter}")
    else:
        frozen_max_iter = None
        print("\n[6/9] Skipping early stopping (not applicable to tree ensembles).")

    print(f"\n[7/9] Fitting final {best_clf_name} model on all {len(train_df):,} training samples...")
    w_outer_train = compute_sample_weights(y_train)

    if best_clf_name == "HistGradientBoosting":
        final_model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=best_params["learning_rate"],
            max_leaf_nodes=best_params["max_leaf_nodes"],
            min_samples_leaf=best_params["min_samples_leaf"],
            l2_regularization=best_params["l2_regularization"],
            max_iter=frozen_max_iter,
            max_bins=255,
            categorical_features=cat_indices,
            early_stopping=False,
            random_state=42,
        )
        final_model.fit(X_train_encoded, y_train, sample_weight=w_outer_train)
    elif best_clf_name == "RandomForest":
        final_model = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        final_model.fit(X_train_encoded, y_train)
    else:
        final_model = ExtraTreesClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        final_model.fit(X_train_encoded, y_train)

    print("\n[8/9] Evaluating final model on held-out spatial blocks...")
    y_test_proba = final_model.predict_proba(X_test_encoded)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    pr_auc = float(average_precision_score(y_test, y_test_proba))
    prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    p80 = float(np.percentile(y_test_proba, 80))
    p95 = float(np.percentile(y_test_proba, 95))

    test_pos_proba = y_test_proba[y_test == 1]
    test_unl_proba = y_test_proba[y_test == 0]
    train_proba = final_model.predict_proba(X_train_encoded)[:, 1]
    train_pos_proba = train_proba[y_train == 1]
    train_unl_proba = train_proba[y_train == 0]

    print("=" * 60)
    print("   FINAL MODEL METRICS (SPATIAL HOLDOUT)")
    print("=" * 60)
    print(f"   Classifier:         {best_clf_name}")
    print(f"   Features:           {len(feature_cols)} ({schema['numeric_feature_count']} numeric + {schema['categorical_feature_count']} categorical)")
    print(f"   ROC-AUC Score:      {roc_auc:.4f}")
    print(f"   PR-AUC Score:       {pr_auc:.4f}")
    print(f"   Precision:          {prec:.4f}")
    print(f"   Recall:             {rec:.4f}")
    print(f"   F1-Score:           {f1:.4f}")
    print(f"   Balanced Accuracy:  {bal_acc:.4f}")
    print(f"   Confusion Matrix:   TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"   Empirical Cutoffs:  Moderate (p80)={p80:.4f}, High (p95)={p95:.4f}")
    print()
    print("   SCORE DISTRIBUTIONS:")
    print(f"   Train POS median:   {np.median(train_pos_proba):.4f}")
    print(f"   Train UNL median:   {np.median(train_unl_proba):.4f}")
    print(f"   Test POS median:    {np.median(test_pos_proba):.4f}")
    print(f"   Test UNL median:    {np.median(test_unl_proba):.4f}")
    print(f"   Discrimination:     {np.median(test_pos_proba)/max(np.median(test_unl_proba),1e-9):.1f}x")
    print("=" * 60)

    print("\nComputing permutation feature importances (20 repeats)...")
    perm = permutation_importance(
        final_model, X_test_encoded, y_test,
        scoring="average_precision", n_repeats=20,
        random_state=42, n_jobs=-1,
    )

    importances = perm.importances_mean
    importances = np.maximum(importances, 0.0)
    total_imp = np.sum(importances) if np.sum(importances) > 0 else 1.0
    imp_df = pd.DataFrame([
        {
            "feature": feat,
            "domain": get_feature_domain(feat),
            "importance": round(float(imp), 6),
            "importance_pct": round(float(imp / total_imp * 100), 3),
        }
        for feat, imp in zip(feature_cols, importances)
    ]).sort_values(by="importance", ascending=False).reset_index(drop=True)

    imp_file = os.path.join(base_dir, "feature_importance.csv")
    imp_df.to_csv(imp_file, index=False)
    print(f"Saved feature importances to {imp_file}")
    print("\nTop 15 features by permutation importance:")
    for _, row in imp_df.head(15).iterrows():
        print(f"   {row['feature']:<35s} {row['domain']:<30s} {row['importance_pct']:>7.2f}%")

    print("\nFitting IsolationForest applicability model on training features...")
    iso_forest = IsolationForest(random_state=42, contamination="auto", n_jobs=-1)
    iso_forest.fit(X_train_encoded.fillna(0.0))
    train_scores = iso_forest.decision_function(X_train_encoded.fillna(0.0))
    p20_applicability = float(np.percentile(train_scores, 20))
    p05_applicability = float(np.percentile(train_scores, 5))
    print(f"Applicability cutoffs: Moderate={p20_applicability:.4f}, Low={p05_applicability:.4f}")

    print("\n[9/9] Building geological lookup grid for inference...")
    geo_cols = [
        'geological_group', 'geological_formation', 'stratigraphic_unit',
        'lithology', 'metamorphic_grade', 'weathering_class',
        'fold_position', 'structural_orientation',
    ]
    geo_lookup_df = clean_df[['latitude', 'longitude'] + geo_cols].drop_duplicates(
        subset=['latitude', 'longitude']
    ).reset_index(drop=True)
    geo_lookup_path = os.path.join(base_dir, "geological_lookup_grid.csv")
    geo_lookup_df.to_csv(geo_lookup_path, index=False)
    print(f"Saved geological lookup grid: {len(geo_lookup_df)} unique locations to {geo_lookup_path}")

    pos_mask = train_df["target"] == 1
    unl_mask = train_df["target"] == 0
    s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    s2_positive_mean = train_df.loc[pos_mask, s2_bands].mean().to_dict()
    s2_unlabelled_mean = train_df.loc[unl_mask, s2_bands].mean().to_dict()
    radar_features = ['VV', 'VH', 'VV_VH_ratio', 'radar_backscatter_mean', 'radar_texture']
    radar_positive_mean = train_df.loc[pos_mask, radar_features].mean().to_dict()

    model_bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_indices": cat_indices,
        "cat_mappings": cat_mappings,
        "classifier_name": best_clf_name,
        "schema_version": schema["schema_version"],
        "domain_bbox": schema["study_domain"],
        "probability_thresholds": {
            "moderate_cutoff": p80,
            "high_cutoff": p95,
            "description": "Empirically-derived percentile thresholds (80th and 95th percentiles of spatial test predictions). Prototype ranking cutoffs, not field-calibrated probabilities.",
        },
        "applicability": {
            "model": iso_forest,
            "threshold_moderate": p20_applicability,
            "threshold_low": p05_applicability,
            "description": "IsolationForest decision_function thresholds (20th and 5th percentiles of training feature distribution).",
        },
        "derived_constants": schema["derived_constants"],
        "s2_positive_mean": s2_positive_mean,
        "s2_unlabelled_mean": s2_unlabelled_mean,
        "radar_positive_mean": radar_positive_mean,
        "classifier_comparison": comparison_results,
        "metadata": {
            "model_type": best_clf_name,
            "loss": "log_loss" if best_clf_name == "HistGradientBoosting" else "gini",
            "features_count": len(feature_cols),
            "study_area": schema["study_domain"]["region_name"],
            "frozen_max_iter": frozen_max_iter,
            "winning_hyperparameters": best_params,
            "cv_mean_pr_auc": round(best_res["mean_pr_auc"], 4),
            "cv_std_pr_auc": round(best_res.get("std_pr_auc", 0), 4),
        },
        "score_distributions": {
            "train_pos_median": round(float(np.median(train_pos_proba)), 4),
            "train_unl_median": round(float(np.median(train_unl_proba)), 4),
            "test_pos_median": round(float(np.median(test_pos_proba)), 4),
            "test_unl_median": round(float(np.median(test_unl_proba)), 4),
        },
    }

    model_file = os.path.join(base_dir, "manganese_prospectivity_model.pkl")
    joblib.dump(model_bundle, model_file)
    print(f"\nSaved authoritative model bundle to: {model_file}")

    metrics_payload = {
        "model_name": best_clf_name,
        "features_type": f"{len(feature_cols)} features (satellite + terrain + geological map)",
        "schema_version": schema["schema_version"],
        "n_features": len(feature_cols),
        "validation_strategy": "Spatial GroupShuffleSplit on spatial_block_id (80/20 holdout)",
        "spatial_tuning_strategy": "5-Fold GroupKFold on spatial_block_id",
        "classifier_comparison": comparison_results,
        "winning_hyperparameters": best_params,
        "cv_5fold_pr_auc_mean": round(best_res["mean_pr_auc"], 4),
        "cv_5fold_pr_auc_std": round(best_res.get("std_pr_auc", 0), 4),
        "frozen_max_iter": frozen_max_iter,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "confusion_matrix": {
                "true_unlabelled_correct": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
        },
        "score_distributions": {
            "train_pos_median": round(float(np.median(train_pos_proba)), 4),
            "train_unl_median": round(float(np.median(train_unl_proba)), 4),
            "test_pos_median": round(float(np.median(test_pos_proba)), 4),
            "test_unl_median": round(float(np.median(test_unl_proba)), 4),
        },
        "probability_thresholds": {
            "moderate_cutoff_p80": round(p80, 4),
            "high_cutoff_p95": round(p95, 4),
        },
        "applicability_thresholds": {
            "moderate_applicability_p20": round(p20_applicability, 4),
            "low_applicability_p05": round(p05_applicability, 4),
        },
        "disclaimer": "Spatially validated prototype manganese prospectivity-ranking model trained on satellite, terrain, and geological map features. Not a reserve estimate.",
    }

    metrics_file = os.path.join(base_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
