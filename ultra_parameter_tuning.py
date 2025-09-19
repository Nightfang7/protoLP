#!/usr/bin/env python3
"""
超激進參數調優 - 測試極端參數組合來尋找更高準確率
"""

import os
import sys
from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

# 設定環境變數
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def ultra_parameter_tuning():
    """超激進參數調優實驗"""
    print("開始超激進參數調優實驗...")
    print("測試極端參數組合來尋找更高準確率")
    print("目標：突破88.24%，達到89%+")
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42
    
    # 定義極端參數組合
    param_combinations = [
        # 1. 基於最佳結果的細緻調整
        {'name': 'extreme_1_tuned_1', 'beta': 0.03, 'reduction_dim': 12, 'lam': 0.3, 'alpha': 0.005, 'n_epochs': 300, 'distance_metric': 'euclidean'},
        {'name': 'extreme_1_tuned_2', 'beta': 0.08, 'reduction_dim': 18, 'lam': 0.8, 'alpha': 0.015, 'n_epochs': 250, 'distance_metric': 'euclidean'},
        {'name': 'extreme_1_tuned_3', 'beta': 0.02, 'reduction_dim': 10, 'lam': 0.2, 'alpha': 0.003, 'n_epochs': 400, 'distance_metric': 'euclidean'},
        {'name': 'extreme_1_tuned_4', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 2. 極小Power transformation參數
        {'name': 'beta_0.01', 'beta': 0.01, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.02', 'beta': 0.02, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.05', 'beta': 0.05, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.08', 'beta': 0.08, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.12', 'beta': 0.12, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.15', 'beta': 0.15, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 3. 極小降維維度
        {'name': 'dim_5', 'beta': 0.5, 'reduction_dim': 5, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_8', 'beta': 0.5, 'reduction_dim': 8, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_10', 'beta': 0.5, 'reduction_dim': 10, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_12', 'beta': 0.5, 'reduction_dim': 12, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_15', 'beta': 0.5, 'reduction_dim': 15, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_18', 'beta': 0.5, 'reduction_dim': 18, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 4. 極小正則化參數
        {'name': 'lam_0.1', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.1, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_0.2', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.2, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_0.3', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.3, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_0.4', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.4, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_0.6', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.6, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_0.8', 'beta': 0.5, 'reduction_dim': 40, 'lam': 0.8, 'alpha': 0.2, 'n_epochs': 20},
        
        # 5. 極小MAP優化參數
        {'name': 'alpha_0.001', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.001, 'n_epochs': 20},
        {'name': 'alpha_0.005', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.005, 'n_epochs': 20},
        {'name': 'alpha_0.01', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.01, 'n_epochs': 20},
        {'name': 'alpha_0.02', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.02, 'n_epochs': 20},
        {'name': 'alpha_0.03', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.03, 'n_epochs': 20},
        {'name': 'alpha_0.05', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.05, 'n_epochs': 20},
        
        # 6. 極多迭代次數
        {'name': 'epochs_150', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 150},
        {'name': 'epochs_200', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 200},
        {'name': 'epochs_300', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 300},
        {'name': 'epochs_400', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 400},
        {'name': 'epochs_500', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 500},
        
        # 7. 超激進組合
        {'name': 'ultra_1', 'beta': 0.01, 'reduction_dim': 8, 'lam': 0.1, 'alpha': 0.001, 'n_epochs': 500, 'distance_metric': 'euclidean'},
        {'name': 'ultra_2', 'beta': 0.02, 'reduction_dim': 10, 'lam': 0.2, 'alpha': 0.002, 'n_epochs': 400, 'distance_metric': 'cosine'},
        {'name': 'ultra_3', 'beta': 0.03, 'reduction_dim': 12, 'lam': 0.3, 'alpha': 0.003, 'n_epochs': 300, 'distance_metric': 'euclidean'},
        {'name': 'ultra_4', 'beta': 0.05, 'reduction_dim': 15, 'lam': 0.5, 'alpha': 0.005, 'n_epochs': 250, 'distance_metric': 'cosine'},
        {'name': 'ultra_5', 'beta': 0.08, 'reduction_dim': 18, 'lam': 0.8, 'alpha': 0.008, 'n_epochs': 200, 'distance_metric': 'euclidean'},
        
        # 8. 關閉所有預處理的組合
        {'name': 'no_preprocessing_1', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_power_transform': False, 'use_unitary_scaling': False, 'use_centering': False},
        {'name': 'no_preprocessing_2', 'beta': 0.1, 'reduction_dim': 20, 'lam': 1, 'alpha': 0.01, 'n_epochs': 100, 'use_power_transform': False, 'use_unitary_scaling': False, 'use_centering': False},
        {'name': 'no_preprocessing_3', 'beta': 0.05, 'reduction_dim': 15, 'lam': 0.5, 'alpha': 0.005, 'n_epochs': 200, 'use_power_transform': False, 'use_unitary_scaling': False, 'use_centering': False},
        
        # 9. 混合策略
        {'name': 'hybrid_1', 'beta': 0.02, 'reduction_dim': 12, 'lam': 0.3, 'alpha': 0.003, 'n_epochs': 300, 'use_power_transform': False, 'distance_metric': 'euclidean'},
        {'name': 'hybrid_2', 'beta': 0.03, 'reduction_dim': 15, 'lam': 0.4, 'alpha': 0.004, 'n_epochs': 250, 'use_unitary_scaling': False, 'distance_metric': 'cosine'},
        {'name': 'hybrid_3', 'beta': 0.04, 'reduction_dim': 18, 'lam': 0.5, 'alpha': 0.005, 'n_epochs': 200, 'use_centering': False, 'distance_metric': 'euclidean'},
        
        # 10. 極端測試
        {'name': 'extreme_test_1', 'beta': 0.005, 'reduction_dim': 5, 'lam': 0.05, 'alpha': 0.0005, 'n_epochs': 1000, 'distance_metric': 'euclidean'},
        {'name': 'extreme_test_2', 'beta': 0.015, 'reduction_dim': 8, 'lam': 0.15, 'alpha': 0.0015, 'n_epochs': 800, 'distance_metric': 'cosine'},
        {'name': 'extreme_test_3', 'beta': 0.025, 'reduction_dim': 10, 'lam': 0.25, 'alpha': 0.0025, 'n_epochs': 600, 'distance_metric': 'euclidean'},
    ]
    
    results = {}
    
    print(f"總共要測試 {len(param_combinations)} 個極端參數組合")
    
    for i, params in enumerate(param_combinations):
        print(f"\n=== 測試參數組合 {i+1}/{len(param_combinations)}: {params['name']} ===")
        print(f"參數: {params}")
        
        try:
            # 移除name參數
            test_params = {k: v for k, v in params.items() if k != 'name'}
            _, avg_results = evaluate_lizz_35_cases_tuned(
                data_root, "00177", "5shots", 5, 
                autoencoder_dir, random_seed, **test_params
            )
            
            results[params['name']] = {
                'params': params,
                'accuracy': avg_results['accuracy'],
                'precision': avg_results['precision'],
                'recall': avg_results['recall'],
                'f1': avg_results['f1'],
                'pr_auc': avg_results['pr_auc']
            }
            
            print(f"結果: Accuracy={avg_results['accuracy']:.4f}, F1={avg_results['f1']:.4f}, PR-AUC={avg_results['pr_auc']:.4f}")
            
        except Exception as e:
            print(f"參數組合 {params['name']} 失敗: {e}")
            results[params['name']] = {
                'params': params,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'pr_auc': 0.0
            }
    
    # 找出最佳參數
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    best_pr_auc = max(results.items(), key=lambda x: x[1]['pr_auc'])
    
    print(f"\n=== 超激進參數調優結果 ===")
    print(f"最佳準確率: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")
    print(f"最佳F1分數: {best_f1[0]} - {best_f1[1]['f1']:.4f}")
    print(f"最佳PR-AUC: {best_pr_auc[0]} - {best_pr_auc[1]['pr_auc']:.4f}")
    
    # 顯示前15名
    print(f"\n=== 前15名準確率 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:15]):
        print(f"{i+1:2d}. {name:<25} Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}, PR-AUC={result['pr_auc']:.4f}")
    
    # 保存結果
    import pickle
    with open('lizz_evaluation_results/ultra_parameter_tuning_results_00177_5shots_5way.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n詳細結果已保存到: lizz_evaluation_results/ultra_parameter_tuning_results_00177_5shots_5way.pkl")
    
    return results, best_accuracy, best_f1, best_pr_auc

if __name__ == "__main__":
    ultra_parameter_tuning()
