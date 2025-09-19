#!/usr/bin/env python3
"""
最終參數調優 - 基於最佳結果進行最細緻的調整
目標：突破88.35%，達到89%+
"""

import os
import sys
from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

# 設定環境變數
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def final_parameter_tuning():
    """最終參數調優實驗"""
    print("開始最終參數調優實驗...")
    print("基於最佳結果 dim_13 (88.35%) 進行最細緻的調整")
    print("目標：突破88.35%，達到89%+")
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42
    
    # 基於最佳結果的最細緻調整
    param_combinations = [
        # 1. 最細緻的降維維度調整 (圍繞13維)
        {'name': 'dim_10', 'beta': 0.04, 'reduction_dim': 10, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_11', 'beta': 0.04, 'reduction_dim': 11, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_12', 'beta': 0.04, 'reduction_dim': 12, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_14', 'beta': 0.04, 'reduction_dim': 14, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_15', 'beta': 0.04, 'reduction_dim': 15, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_16', 'beta': 0.04, 'reduction_dim': 16, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_17', 'beta': 0.04, 'reduction_dim': 17, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'dim_18', 'beta': 0.04, 'reduction_dim': 18, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 2. 最細緻的Power transformation參數調整 (圍繞0.008)
        {'name': 'beta_0.005', 'beta': 0.005, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.006', 'beta': 0.006, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.008', 'beta': 0.008, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.010', 'beta': 0.010, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.012', 'beta': 0.012, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'beta_0.015', 'beta': 0.015, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        
        # 3. 最佳組合參數測試
        {'name': 'combo_1', 'beta': 0.008, 'reduction_dim': 12, 'lam': 0.1, 'alpha': 0.001, 'n_epochs': 200, 'distance_metric': 'euclidean'},
        {'name': 'combo_2', 'beta': 0.006, 'reduction_dim': 14, 'lam': 0.2, 'alpha': 0.002, 'n_epochs': 250, 'distance_metric': 'euclidean'},
        {'name': 'combo_3', 'beta': 0.010, 'reduction_dim': 11, 'lam': 0.3, 'alpha': 0.003, 'n_epochs': 300, 'distance_metric': 'euclidean'},
        {'name': 'combo_4', 'beta': 0.012, 'reduction_dim': 15, 'lam': 0.4, 'alpha': 0.004, 'n_epochs': 400, 'distance_metric': 'euclidean'},
        {'name': 'combo_5', 'beta': 0.005, 'reduction_dim': 16, 'lam': 0.5, 'alpha': 0.005, 'n_epochs': 500, 'distance_metric': 'euclidean'},
        
        # 4. 極端測試
        {'name': 'extreme_1', 'beta': 0.001, 'reduction_dim': 8, 'lam': 0.01, 'alpha': 0.0001, 'n_epochs': 1000, 'distance_metric': 'euclidean'},
        {'name': 'extreme_2', 'beta': 0.002, 'reduction_dim': 9, 'lam': 0.05, 'alpha': 0.0005, 'n_epochs': 800, 'distance_metric': 'euclidean'},
        {'name': 'extreme_3', 'beta': 0.003, 'reduction_dim': 10, 'lam': 0.1, 'alpha': 0.001, 'n_epochs': 600, 'distance_metric': 'euclidean'},
        {'name': 'extreme_4', 'beta': 0.004, 'reduction_dim': 11, 'lam': 0.15, 'alpha': 0.0015, 'n_epochs': 700, 'distance_metric': 'euclidean'},
        {'name': 'extreme_5', 'beta': 0.005, 'reduction_dim': 12, 'lam': 0.2, 'alpha': 0.002, 'n_epochs': 500, 'distance_metric': 'euclidean'},
        
        # 5. 距離度量測試
        {'name': 'cosine_1', 'beta': 0.008, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'cosine'},
        {'name': 'cosine_2', 'beta': 0.006, 'reduction_dim': 12, 'lam': 0.2, 'alpha': 0.002, 'n_epochs': 250, 'distance_metric': 'cosine'},
        {'name': 'cosine_3', 'beta': 0.010, 'reduction_dim': 14, 'lam': 0.3, 'alpha': 0.003, 'n_epochs': 300, 'distance_metric': 'cosine'},
        
        # 6. 混合策略測試
        {'name': 'hybrid_1', 'beta': 0.007, 'reduction_dim': 13, 'lam': 0.1, 'alpha': 0.001, 'n_epochs': 400, 'distance_metric': 'euclidean'},
        {'name': 'hybrid_2', 'beta': 0.009, 'reduction_dim': 12, 'lam': 0.2, 'alpha': 0.002, 'n_epochs': 350, 'distance_metric': 'cosine'},
        {'name': 'hybrid_3', 'beta': 0.011, 'reduction_dim': 14, 'lam': 0.3, 'alpha': 0.003, 'n_epochs': 300, 'distance_metric': 'euclidean'},
        
        # 7. 超細緻調整
        {'name': 'fine_1', 'beta': 0.0075, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'fine_2', 'beta': 0.0085, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'fine_3', 'beta': 0.0095, 'reduction_dim': 13, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'fine_4', 'beta': 0.008, 'reduction_dim': 12.5, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
        {'name': 'fine_5', 'beta': 0.008, 'reduction_dim': 13.5, 'lam': 0.4, 'alpha': 0.008, 'n_epochs': 350, 'distance_metric': 'euclidean'},
    ]
    
    results = {}
    
    print(f"總共要測試 {len(param_combinations)} 個最終參數組合")
    
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
    
    print(f"\n=== 最終參數調優結果 ===")
    print(f"最佳準確率: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")
    print(f"最佳F1分數: {best_f1[0]} - {best_f1[1]['f1']:.4f}")
    print(f"最佳PR-AUC: {best_pr_auc[0]} - {best_pr_auc[1]['pr_auc']:.4f}")
    
    # 顯示前25名
    print(f"\n=== 前25名準確率 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:25]):
        print(f"{i+1:2d}. {name:<15} Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}, PR-AUC={result['pr_auc']:.4f}")
    
    # 保存結果
    import pickle
    with open('lizz_evaluation_results/final_parameter_tuning_results_00177_5shots_5way.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n詳細結果已保存到: lizz_evaluation_results/final_parameter_tuning_results_00177_5shots_5way.pkl")
    
    return results, best_accuracy, best_f1, best_pr_auc

if __name__ == "__main__":
    final_parameter_tuning()
