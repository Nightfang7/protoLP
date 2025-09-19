#!/usr/bin/env python3
"""
快速讀取pkl結果檔案的簡單工具
"""

import pickle
import os
import glob

def quick_read_pkl(pkl_path):
    """
    快速讀取並顯示pkl檔案內容
    
    Args:
        pkl_path (str): pkl檔案路徑
    """
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"檔案: {os.path.basename(pkl_path)}")
        print("="*50)
        
        # 顯示配置信息
        config = results.get('config', {})
        print("配置信息:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 顯示平均結果
        avg_metrics = results.get('average_metrics', {})
        if avg_metrics:
            print("\n平均結果 (35個Cases):")
            print(f"  準確率: {avg_metrics.get('avg_accuracy', 0):.4f} ± {avg_metrics.get('std_accuracy', 0):.4f}")
            print(f"  精確率: {avg_metrics.get('avg_precision', 0):.4f} ± {avg_metrics.get('std_precision', 0):.4f}")
            print(f"  召回率: {avg_metrics.get('avg_recall', 0):.4f} ± {avg_metrics.get('std_recall', 0):.4f}")
            print(f"  F1分數: {avg_metrics.get('avg_f1_score', 0):.4f} ± {avg_metrics.get('std_f1_score', 0):.4f}")
            print(f"  PR-AUC: {avg_metrics.get('avg_pr_auc', 0):.4f} ± {avg_metrics.get('std_pr_auc', 0):.4f}")
        
        # 顯示前5個case的結果
        individual_cases = results.get('individual_cases', [])
        if individual_cases:
            print(f"\n前5個Case結果:")
            print("Case  Accuracy  Precision  Recall    F1-Score  PR-AUC")
            print("-" * 55)
            for i, case in enumerate(individual_cases[:5], 1):
                print(f"{i:4d}  {case.get('accuracy', 0):8.4f}  {case.get('precision', 0):9.4f}  "
                      f"{case.get('recall', 0):8.4f}  {case.get('f1_score', 0):8.4f}  {case.get('pr_auc', 0):6.4f}")
            if len(individual_cases) > 5:
                print(f"... 還有 {len(individual_cases) - 5} 個cases")
        
        print("\n")
        
    except Exception as e:
        print(f"讀取檔案失敗: {e}")

def list_all_results(results_dir="lizz_evaluation_results"):
    """
    列出所有結果檔案
    
    Args:
        results_dir (str): 結果目錄
    """
    if not os.path.exists(results_dir):
        print(f"結果目錄不存在: {results_dir}")
        return
    
    pkl_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"在 {results_dir} 中找不到pkl檔案")
        return
    
    print(f"找到 {len(pkl_files)} 個結果檔案:")
    for i, pkl_file in enumerate(sorted(pkl_files), 1):
        print(f"{i:2d}. {os.path.basename(pkl_file)}")

def main():
    """
    主函數
    """
    print("快速pkl結果讀取工具")
    print("="*30)
    
    # 列出所有結果檔案
    list_all_results()
    
    # 讀取所有結果檔案
    results_dir = "lizz_evaluation_results"
    pkl_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if pkl_files:
        print(f"\n讀取所有結果檔案:")
        for pkl_file in sorted(pkl_files):
            quick_read_pkl(pkl_file)

if __name__ == "__main__":
    main()
