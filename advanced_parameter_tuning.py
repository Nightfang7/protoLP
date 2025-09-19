#!/usr/bin/env python3
"""
進階參數調優 - 測試更廣泛的參數範圍
"""

import os
import sys
from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

# 設定環境變數
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def advanced_parameter_tuning():
    """進階參數調優實驗"""
    print("開始進階參數調優實驗...")
    print("測試更廣泛的參數範圍來尋找更高準確率")
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42
    
    # 定義更廣泛的參數組合
    param_combinations = [
        # 1. 更細緻的Power transformation參數
        {'name': 'beta_0.1', 'beta': 0.1, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.2', 'beta': 0.2, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.4', 'beta': 0.4, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.6', 'beta': 0.6, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.8', 'beta': 0.8, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.9', 'beta': 0.9, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_1.2', 'beta': 1.2, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_1.5', 'beta': 1.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 2. 更廣泛的降維維度
        {'name': 'dim_20', 'beta': 0.5, 'reduction_dim': 20, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_25', 'beta': 0.5, 'reduction_dim': 25, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_35', 'beta': 0.5, 'reduction_dim': 35, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_45', 'beta': 0.5, 'reduction_dim': 45, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_55', 'beta': 0.5, 'reduction_dim': 55, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_70', 'beta': 0.5, 'reduction_dim': 70, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_80', 'beta': 0.5, 'reduction_dim': 80, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 3. 更廣泛的正則化參數
        {'name': 'lam_1', 'beta': 0.5, 'reduction_dim': 40, 'lam': 1, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_2', 'beta': 0.5, 'reduction_dim': 40, 'lam': 2, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_3', 'beta': 0.5, 'reduction_dim': 40, 'lam': 3, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_8', 'beta': 0.5, 'reduction_dim': 40, 'lam': 8, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_15', 'beta': 0.5, 'reduction_dim': 40, 'lam': 15, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_25', 'beta': 0.5, 'reduction_dim': 40, 'lam': 25, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_30', 'beta': 0.5, 'reduction_dim': 40, 'lam': 30, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_100', 'beta': 0.5, 'reduction_dim': 40, 'lam': 100, 'alpha': 0.2, 'n_epochs': 20},
        
        # 4. 更廣泛的MAP優化參數
        {'name': 'alpha_0.05', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.05, 'n_epochs': 20},
        {'name': 'alpha_0.15', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.15, 'n_epochs': 20},
        {'name': 'alpha_0.25', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.25, 'n_epochs': 20},
        {'name': 'alpha_0.35', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.35, 'n_epochs': 20},
        {'name': 'alpha_0.4', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.4, 'n_epochs': 20},
        {'name': 'alpha_0.6', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.6, 'n_epochs': 20},
        {'name': 'alpha_0.8', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.8, 'n_epochs': 20},
        {'name': 'alpha_1.0', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 1.0, 'n_epochs': 20},
        
        # 5. 更多迭代次數
        {'name': 'epochs_10', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 10},
        {'name': 'epochs_30', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 30},
        {'name': 'epochs_40', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 40},
        {'name': 'epochs_60', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 60},
        {'name': 'epochs_80', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 80},
        {'name': 'epochs_100', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 100},
        
        # 6. 組合最佳參數的變體
        {'name': 'combo_aggressive_1', 'beta': 0.2, 'reduction_dim': 70, 'lam': 2, 'alpha': 0.05, 'n_epochs': 80, 'distance_metric': 'euclidean'},
        {'name': 'combo_aggressive_2', 'beta': 0.4, 'reduction_dim': 25, 'lam': 1, 'alpha': 0.15, 'n_epochs': 100, 'distance_metric': 'cosine'},
        {'name': 'combo_aggressive_3', 'beta': 0.1, 'reduction_dim': 80, 'lam': 3, 'alpha': 0.1, 'n_epochs': 60, 'distance_metric': 'euclidean'},
        {'name': 'combo_aggressive_4', 'beta': 0.6, 'reduction_dim': 30, 'lam': 5, 'alpha': 0.25, 'n_epochs': 40, 'distance_metric': 'cosine'},
        
        # 7. 關閉多個預處理步驟的組合
        {'name': 'no_power_no_centering', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_power_transform': False, 'use_centering': False},
        {'name': 'no_scaling_no_centering', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_unitary_scaling': False, 'use_centering': False},
        {'name': 'no_power_no_scaling', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_power_transform': False, 'use_unitary_scaling': False},
        {'name': 'minimal_preprocessing', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_power_transform': False, 'use_unitary_scaling': False, 'use_centering': False},
        
        # 8. 極端參數組合
        {'name': 'extreme_1', 'beta': 0.05, 'reduction_dim': 15, 'lam': 0.5, 'alpha': 0.01, 'n_epochs': 200, 'distance_metric': 'euclidean'},
        {'name': 'extreme_2', 'beta': 2.0, 'reduction_dim': 90, 'lam': 200, 'alpha': 2.0, 'n_epochs': 5, 'distance_metric': 'cosine'},
    ]
    
    results = {}
    
    print(f"總共要測試 {len(param_combinations)} 個參數組合")
    
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
    
    print(f"\n=== 進階參數調優結果 ===")
    print(f"最佳準確率: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")
    print(f"最佳F1分數: {best_f1[0]} - {best_f1[1]['f1']:.4f}")
    print(f"最佳PR-AUC: {best_pr_auc[0]} - {best_pr_auc[1]['pr_auc']:.4f}")
    
    # 顯示前10名
    print(f"\n=== 前10名準確率 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. {name:<20} Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}, PR-AUC={result['pr_auc']:.4f}")
    
    # 保存結果
    import pickle
    with open('lizz_evaluation_results/advanced_parameter_tuning_results_00177_5shots_5way.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n詳細結果已保存到: lizz_evaluation_results/advanced_parameter_tuning_results_00177_5shots_5way.pkl")
    
    return results, best_accuracy, best_f1, best_pr_auc

if __name__ == "__main__":
    advanced_parameter_tuning()
