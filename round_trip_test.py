import os
import pandas as pd
import numpy as np
from predict import predict_single_location, load_model_bundle

def main():
    print("=" * 80)
    print("   ROUND-TRIP VERIFICATION TEST")
    print("   CSV Row → Feature Extraction → Preprocessing → Model → Prediction")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "dataset", "sausar_manganese_core_features.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "sausar_manganese_core_features.csv")

    df = pd.read_csv(data_path)
    bundle = load_model_bundle()
    model_features = bundle.get("feature_cols", bundle.get("MODEL_FEATURES", []))

    print(f"\nDataset: {len(df)} rows")
    print(f"Model features: {len(model_features)}")
    print(f"Model features list: {model_features}")

    pos_df = df[df['label_status'] == 'POSITIVE'].copy()
    unl_df = df[df['label_status'] == 'UNLABELLED'].copy()
    uncertain_df = df[df['label_status'] == 'UNCERTAIN'].copy()

    print(f"\nPOSITIVE: {len(pos_df)} rows")
    print(f"UNLABELLED: {len(unl_df)} rows")
    print(f"UNCERTAIN (excluded): {len(uncertain_df)} rows")

    print("\n" + "=" * 80)
    print("   POSITIVE ROWS ROUND-TRIP TEST")
    print("=" * 80)

    pos_probs = []
    pos_results = []

    for idx, row in pos_df.iterrows():

        row_dict = row.to_dict()
        result = predict_single_location(row_dict, bundle)
        prob = result.get('prospectivity_probability')
        cls = result.get('classification')
        val = result.get('feature_validation', {})

        pos_probs.append(prob)
        pos_results.append({
            'index': idx,
            'lat': row['latitude'],
            'lon': row['longitude'],
            'label': 'POSITIVE',
            'probability': prob,
            'classification': cls,
            'missing_features': len(val.get('missing_features', [])),
            'received_features': val.get('received_count', 0)
        })

        print(f"Index {idx:>6} | lat {row['latitude']:.4f} | lon {row['longitude']:.4f} | "
              f"label POSITIVE | prob {prob:.4f} | cat {cls} | "
              f"features: {val.get('received_count', 0)}/{val.get('expected_count', 0)} "
              f"(missing: {len(val.get('missing_features', []))})")

    print("\n" + "=" * 80)
    print("   UNLABELLED ROWS ROUND-TRIP TEST (sample of 200)")
    print("=" * 80)

    unl_sample = unl_df.sample(n=min(200, len(unl_df)), random_state=42)
    unl_probs = []
    unl_results = []

    for idx, row in unl_sample.iterrows():
        row_dict = row.to_dict()
        result = predict_single_location(row_dict, bundle)
        prob = result.get('prospectivity_probability')
        cls = result.get('classification')
        val = result.get('feature_validation', {})

        unl_probs.append(prob)
        unl_results.append({
            'index': idx,
            'lat': row['latitude'],
            'lon': row['longitude'],
            'label': 'UNLABELLED',
            'probability': prob,
            'classification': cls,
            'missing_features': len(val.get('missing_features', [])),
            'received_features': val.get('received_count', 0)
        })

        print(f"Index {idx:>6} | lat {row['latitude']:.4f} | lon {row['longitude']:.4f} | "
              f"label UNLABELLED | prob {prob:.4f} | cat {cls} | "
              f"features: {val.get('received_count', 0)}/{val.get('expected_count', 0)} "
              f"(missing: {len(val.get('missing_features', []))})")

    print("\n" + "=" * 80)
    print("   STATISTICAL SUMMARY")
    print("=" * 80)

    pos_probs = np.array(pos_probs)
    unl_probs = np.array(unl_probs)

    print(f"\nPOSITIVE distribution (n={len(pos_probs)}):")
    print(f"  Mean:   {pos_probs.mean():.4f}")
    print(f"  Median: {np.median(pos_probs):.4f}")
    print(f"  Std:    {pos_probs.std():.4f}")
    print(f"  Min:    {pos_probs.min():.4f}")
    print(f"  Max:    {pos_probs.max():.4f}")
    print(f"  >0.5:   {(pos_probs > 0.5).sum()} ({(pos_probs > 0.5).mean()*100:.1f}%)")
    print(f"  >0.65:  {(pos_probs > 0.65).sum()} ({(pos_probs > 0.65).mean()*100:.1f}%)")

    print(f"\nUNLABELLED distribution (n={len(unl_probs)}):")
    print(f"  Mean:   {unl_probs.mean():.4f}")
    print(f"  Median: {np.median(unl_probs):.4f}")
    print(f"  Std:    {unl_probs.std():.4f}")
    print(f"  Min:    {unl_probs.min():.4f}")
    print(f"  Max:    {unl_probs.max():.4f}")
    print(f"  >0.5:   {(unl_probs > 0.5).sum()} ({(unl_probs > 0.5).mean()*100:.1f}%)")
    print(f"  >0.65:  {(unl_probs > 0.65).sum()} ({(unl_probs > 0.65).mean()*100:.1f}%)")

    separation = pos_probs.mean() - unl_probs.mean()
    print(f"\nSeparation (POSITIVE mean - UNLABELLED mean): {separation:.4f}")

    pos_high = sum(1 for r in pos_results if r['classification'] == 'HIGH')
    pos_mod = sum(1 for r in pos_results if r['classification'] == 'MODERATE')
    pos_low = sum(1 for r in pos_results if r['classification'] == 'LOW')

    unl_high = sum(1 for r in unl_results if r['classification'] == 'HIGH')
    unl_mod = sum(1 for r in unl_results if r['classification'] == 'MODERATE')
    unl_low = sum(1 for r in unl_results if r['classification'] == 'LOW')

    print(f"\nPOSITIVE classifications: HIGH={pos_high}, MODERATE={pos_mod}, LOW={pos_low}")
    print(f"UNLABELLED classifications: HIGH={unl_high}, MODERATE={unl_mod}, LOW={unl_low}")

    print("\n" + "=" * 80)
    print("   OUT-OF-DOMAIN TEST")
    print("=" * 80)

    ood_tests = [
        {"latitude": 19.0760, "longitude": 72.8777, "name": "Mumbai"},
        {"latitude": 28.6139, "longitude": 77.2090, "name": "Delhi"},
        {"latitude": 12.9716, "longitude": 77.5946, "name": "Bangalore"},
    ]

    for test in ood_tests:
        result = predict_single_location(test, bundle)
        print(f"  {test['name']} ({test['latitude']}, {test['longitude']}): {result['status']} - {result['classification']}")

    print("\n" + "=" * 80)
    print("   WEBSITE FEATURE PIPELINE TEST")
    print("=" * 80)

    test_coords = [
        (21.8900, 80.2200, "Balaghat Mine"),
        (21.6850, 79.7120, "Tirodi Mine"),
        (21.5540, 79.6880, "Dongri Buzurg"),
        (21.2200, 79.4500, "South Bhandara (outside mineralization)"),
    ]

    import feature_pipeline
    for lat, lon, name in test_coords:
        print(f"\n  --- {name} ({lat}, {lon}) ---")
        try:
            val_result = predict_single_location({"latitude": lat, "longitude": lon}, bundle)
            print(f"  Status: {val_result.get('status')}")
            print(f"  Prediction: {val_result.get('prospectivity_percentage')} ({val_result.get('classification')})")
            print(f"  Applicability: {val_result.get('applicability_status')}")
        except Exception as e:
            print(f"  Evaluation error: {e}")

    print("\n" + "=" * 80)
    print("   ROUND-TRIP TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
