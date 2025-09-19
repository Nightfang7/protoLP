#!/usr/bin/env python3
"""
顯示參數調優結果
"""

import pickle

# 讀取結果
with open('lizz_evaluation_results/parameter_tuning_results_00177_5shots_5way.pkl', 'rb') as f:
    results = pickle.load(f)

print("完整參數調優結果 (包含PR-AUC):")
print("="*90)
print(f"{'參數組合':<15} {'準確率':<8} {'F1分數':<8} {'精確率':<8} {'召回率':<8} {'PR-AUC':<8}")
print("-"*90)

# 按準確率排序
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for name, result in sorted_results:
    print(f"{name:<15} {result['accuracy']:.4f}   {result['f1']:.4f}   {result['precision']:.4f}   {result['recall']:.4f}   {result['pr_auc']:.4f}")

print("\n最佳參數組合:")
best = sorted_results[0]
print(f"名稱: {best[0]}")
print(f"參數: {best[1]['params']}")
print(f"準確率: {best[1]['accuracy']:.4f}")
print(f"F1分數: {best[1]['f1']:.4f}")
print(f"PR-AUC: {best[1]['pr_auc']:.4f}")

# 找出最佳PR-AUC
best_pr_auc = max(results.items(), key=lambda x: x[1]['pr_auc'])
print(f"\n最佳PR-AUC參數組合:")
print(f"名稱: {best_pr_auc[0]}")
print(f"PR-AUC: {best_pr_auc[1]['pr_auc']:.4f}")
print(f"準確率: {best_pr_auc[1]['accuracy']:.4f}")
