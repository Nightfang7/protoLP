#!/usr/bin/env python3
"""
使用預設參數批量評估不同資料集與way/shot組合
(defaults from test_lizz_data_evaluation_unbalanced_tuned)
"""

import os
import pickle
from itertools import product
from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42

    # 預設參數（原始defaults）
    default_params = {
        'beta': 0.5,
        'reduction_dim': 40,
        'lam': 10,
        'alpha': 0.2,
        'n_epochs': 20,
        'use_power_transform': True,
        'use_unitary_scaling': True,
        'use_centering': True,
        'use_map_optimization': False,
        'distance_metric': 'euclidean',
    }

    datasets = ["00177", "apidms"]
    shots = ["5shots", "10shots"]
    ways = [5, 10]

    results = {}

    for dataset_name, shot, way in product(datasets, shots, ways):
        print(f"\n=== 預設參數評估: {dataset_name}, {shot}, {way}-way ===")
        try:
            case_results, avg_results = evaluate_lizz_35_cases_tuned(
                data_root, dataset_name, shot, way, autoencoder_dir, random_seed, **default_params
            )
            key = f"{dataset_name}_{shot}_{way}way"
            results[key] = {
                'avg': avg_results,
                'cases': case_results,
                'params': default_params,
            }
            print(f"平均: Acc={avg_results['accuracy']:.4f}, F1={avg_results['f1']:.4f}, PR-AUC={avg_results['pr_auc']:.4f}")
        except Exception as e:
            print(f"組合失敗: {dataset_name}, {shot}, {way}-way → {e}")

    os.makedirs('lizz_evaluation_results', exist_ok=True)
    out_path = 'lizz_evaluation_results/default_params_across_sets.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n全部結果已保存: {out_path}")

if __name__ == "__main__":
    main()
