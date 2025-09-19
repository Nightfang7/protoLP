#!/usr/bin/env python3
"""
針對性參數調優 - 基於最佳結果進行細緻調整
"""

import os
import sys
from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

# 設定環境變數
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def targeted_parameter_tuning():
    """針對性參數調優實驗"""
    print("開始針對性參數調優實驗...")
    print("基於最佳結果 extreme_1_tuned_4 (88.31%) 進行細緻調整")
    print("目標：突破88.31%，達到89%+")
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42
    
    # 基於最佳結果的細緻調整
    param_combinations = [
        # 1. 細緻調整Power transformation參數 (beta)
        {'name': 'beta_0.001', 'beta': 0.001, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.002', 'beta': 0.002, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.003', 'beta': 0.003, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.006', 'beta': 0.006, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.007', 'beta': 0.007, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.009', 'beta': 0.009, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 2. 細緻調整降維維度 (reduction_dim)
        {'name': 'dim_6', 'beta': 0.04, 'reduction_dim': 6, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_7', 'beta': 0.04, 'reduction_dim': 7, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_9', 'beta': 0.04, 'reduction_dim': 9, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_11', 'beta': 0.04, 'reduction_dim': 11, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_13', 'beta': 0.04, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_16', 'beta': 0.04, 'reduction_dim': 16, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_17', 'beta': 0.04, 'reduction_dim': 17, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 3. 細緻調整正則化參數 (lam)
        {'name': 'lam_0.05', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.05, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.1', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.1, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.15', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.15, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.25', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.25, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.35', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.35, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.45', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.45, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'lam_0.6', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.6, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 4. 細緻調整學習率 (alpha)
        {'name': 'alpha_0.0001', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.0001, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.0005', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.0005, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.001', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.001, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.002', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.002, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.003', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.003, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.004', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.004, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.006', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.006, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.007', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.007, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'alpha_0.009', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.009, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 5. 細緻調整迭代次數 (n_epochs)
        {'name': 'epochs_150', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 150, 'distance_metric': 'euclidean'},
        {'name': 'epochs_250', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 250, 'distance_metric': 'euclidean'},
        {'name': 'epochs_300', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 300, 'distance_metric': 'euclidean'},
        {'name': 'epochs_400', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 400, 'distance_metric': 'euclidean'},
        {'name': 'epochs_450', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 450, 'distance_metric': 'euclidean'},
        {'name': 'epochs_500', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 500, 'distance_metric': 'euclidean'},
        {'name': 'epochs_600', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 600, 'distance_metric': 'euclidean'},
        {'name': 'epochs_700', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 700, 'distance_metric': 'euclidean'},
        {'name': 'epochs_800', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 800, 'distance_metric': 'euclidean'},
        
        # 6. 組合最佳參數的變體
        {'name': 'combo_1', 'beta': 0.002, 'reduction_dim': 11, 'lam': 0.15, 'alpha': 0.001, 'n_epochs': 500, 'distance_metric': 'euclidean'},
        {'name': 'combo_2', 'beta': 0.003, 'reduction_dim': 13, 'lam': 0.25, 'alpha': 0.002, 'n_epochs': 450, 'distance_metric': 'euclidean'},
        {'name': 'combo_3', 'beta': 0.006, 'reduction_dim': 16, 'lam': 0.35, 'alpha': 0.004, 'n_epochs': 400, 'distance_metric': 'euclidean'},
        {'name': 'combo_4', 'beta': 0.007, 'reduction_dim': 9, 'lam': 0.1, 'alpha': 0.003, 'n_epochs': 600, 'distance_metric': 'euclidean'},
        {'name': 'combo_5', 'beta': 0.009, 'reduction_dim': 17, 'lam': 0.45, 'alpha': 0.006, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 7. 極端測試
        {'name': 'extreme_1', 'beta': 0.0005, 'reduction_dim': 5, 'lam': 0.05, 'alpha': 0.0001, 'n_epochs': 1000, 'distance_metric': 'euclidean'},
        {'name': 'extreme_2', 'beta': 0.0015, 'reduction_dim': 8, 'lam': 0.1, 'alpha': 0.0005, 'n_epochs': 800, 'distance_metric': 'euclidean'},
        {'name': 'extreme_3', 'beta': 0.0025, 'reduction_dim': 12, 'lam': 0.2, 'alpha': 0.0015, 'n_epochs': 700, 'distance_metric': 'euclidean'},
    ]
    
    results = {}
    
    print(f"總共要測試 {len(param_combinations)} 個針對性參數組合")
    
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
    
    print(f"\n=== 針對性參數調優結果 ===")
    print(f"最佳準確率: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")
    print(f"最佳F1分數: {best_f1[0]} - {best_f1[1]['f1']:.4f}")
    print(f"最佳PR-AUC: {best_pr_auc[0]} - {best_pr_auc[1]['pr_auc']:.4f}")
    
    # 顯示前20名
    print(f"\n=== 前20名準確率 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:20]):
        print(f"{i+1:2d}. {name:<20} Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}, PR-AUC={result['pr_auc']:.4f}")
    
    # 保存結果
    import pickle
    with open('lizz_evaluation_results/targeted_parameter_tuning_results_00177_5shots_5way.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n詳細結果已保存到: lizz_evaluation_results/targeted_parameter_tuning_results_00177_5shots_5way.pkl")
    
    return results, best_accuracy, best_f1, best_pr_auc

if __name__ == "__main__":
    targeted_parameter_tuning()
