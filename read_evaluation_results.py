#!/usr/bin/env python3
"""
評估結果讀取和轉換工具
可以讀取pkl結果檔案並轉換為CSV和TXT格式
"""

import pickle
import pandas as pd
import os
import glob
from datetime import datetime

def read_pkl_results(pkl_path):
    """
    讀取pkl結果檔案
    
    Args:
        pkl_path (str): pkl檔案路徑
    
    Returns:
        dict: 結果字典
    """
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"讀取pkl檔案失敗: {e}")
        return None

def convert_to_csv(results, output_dir="evaluation_results_csv"):
    """
    將結果轉換為CSV格式
    
    Args:
        results (dict): 結果字典
        output_dir (str): 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建詳細結果CSV
    individual_cases = results.get('individual_cases', [])
    if individual_cases:
        # 準備數據
        data = []
        for i, case in enumerate(individual_cases, 1):
            row = {'case_id': i}
            row.update(case)
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"individual_cases_{results['config']['dataset_name']}_{results['config']['shot_type']}_{results['config']['way']}way.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"個別case結果已保存到: {csv_path}")
    
    # 創建平均結果CSV
    avg_metrics = results.get('average_metrics', {})
    if avg_metrics:
        avg_data = []
        config = results.get('config', {})
        
        # 準備平均結果數據
        row = {
            'dataset': config.get('dataset_name', ''),
            'shot_type': config.get('shot_type', ''),
            'way': config.get('way', ''),
            'n_cases': config.get('n_cases', ''),
        }
        
        # 添加所有平均指標
        for key, value in avg_metrics.items():
            if key.startswith('avg_'):
                metric_name = key.replace('avg_', '')
                row[f'{metric_name}_mean'] = value
            elif key.startswith('std_'):
                metric_name = key.replace('std_', '')
                row[f'{metric_name}_std'] = value
        
        avg_data.append(row)
        df_avg = pd.DataFrame(avg_data)
        csv_path = os.path.join(output_dir, f"average_results_{results['config']['dataset_name']}_{results['config']['shot_type']}_{results['config']['way']}way.csv")
        df_avg.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"平均結果已保存到: {csv_path}")

def convert_to_txt(results, output_dir="evaluation_results_txt"):
    """
    將結果轉換為TXT格式
    
    Args:
        results (dict): 結果字典
        output_dir (str): 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config = results.get('config', {})
    avg_metrics = results.get('average_metrics', {})
    individual_cases = results.get('individual_cases', [])
    
    # 創建詳細報告
    txt_path = os.path.join(output_dir, f"detailed_report_{config['dataset_name']}_{config['shot_type']}_{config['way']}way.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Lizz_data Few-Shot Learning 評估結果報告\n")
        f.write("="*80 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"資料集: {config.get('dataset_name', 'N/A')}\n")
        f.write(f"Shot類型: {config.get('shot_type', 'N/A')}\n")
        f.write(f"Way數量: {config.get('way', 'N/A')}\n")
        f.write(f"評估Cases數: {config.get('n_cases', 'N/A')}\n")
        f.write("="*80 + "\n\n")
        
        # 平均結果
        f.write("平均結果 (35個Cases):\n")
        f.write("-" * 50 + "\n")
        if avg_metrics:
            f.write(f"準確率 (Accuracy): {avg_metrics.get('avg_accuracy', 0):.4f} ± {avg_metrics.get('std_accuracy', 0):.4f}\n")
            f.write(f"精確率 (Precision): {avg_metrics.get('avg_precision', 0):.4f} ± {avg_metrics.get('std_precision', 0):.4f}\n")
            f.write(f"召回率 (Recall): {avg_metrics.get('avg_recall', 0):.4f} ± {avg_metrics.get('std_recall', 0):.4f}\n")
            f.write(f"F1分數 (F1-Score): {avg_metrics.get('avg_f1_score', 0):.4f} ± {avg_metrics.get('std_f1_score', 0):.4f}\n")
            f.write(f"PR-AUC: {avg_metrics.get('avg_pr_auc', 0):.4f} ± {avg_metrics.get('std_pr_auc', 0):.4f}\n")
        f.write("\n")
        
        # 個別case結果
        f.write("個別Case結果:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Case':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<12} {'PR-AUC':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, case in enumerate(individual_cases, 1):
            f.write(f"{i:<6} {case.get('accuracy', 0):<10.4f} {case.get('precision', 0):<12.4f} "
                   f"{case.get('recall', 0):<10.4f} {case.get('f1_score', 0):<12.4f} {case.get('pr_auc', 0):<10.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"詳細報告已保存到: {txt_path}")

def process_all_results(results_dir="lizz_evaluation_results"):
    """
    處理所有結果檔案
    
    Args:
        results_dir (str): 結果目錄
    """
    # 尋找所有pkl檔案
    pkl_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"在 {results_dir} 中找不到pkl檔案")
        return
    
    print(f"找到 {len(pkl_files)} 個結果檔案")
    
    # 處理每個檔案
    for pkl_file in pkl_files:
        print(f"\n處理檔案: {os.path.basename(pkl_file)}")
        results = read_pkl_results(pkl_file)
        
        if results:
            convert_to_csv(results)
            convert_to_txt(results)
        else:
            print(f"跳過檔案: {pkl_file}")

def create_summary_report(results_dir="lizz_evaluation_results", output_file="summary_report.txt"):
    """
    創建所有結果的總結報告
    
    Args:
        results_dir (str): 結果目錄
        output_file (str): 輸出檔案名
    """
    pkl_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"在 {results_dir} 中找不到pkl檔案")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Lizz_data Few-Shot Learning 評估結果總結報告\n")
        f.write("="*100 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總共評估了 {len(pkl_files)} 個設定\n\n")
        
        # 總結表格
        f.write("所有設定結果總結:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'設定':<25} {'準確率':<15} {'精確率':<15} {'召回率':<15} {'F1分數':<15} {'PR-AUC':<15}\n")
        f.write("-" * 100 + "\n")
        
        for pkl_file in sorted(pkl_files):
            results = read_pkl_results(pkl_file)
            if results:
                config = results.get('config', {})
                avg_metrics = results.get('average_metrics', {})
                
                dataset = config.get('dataset_name', 'N/A')
                shot_type = config.get('shot_type', 'N/A')
                way = config.get('way', 'N/A')
                setting = f"{dataset}_{shot_type}_{way}way"
                
                accuracy = avg_metrics.get('avg_accuracy', 0)
                precision = avg_metrics.get('avg_precision', 0)
                recall = avg_metrics.get('avg_recall', 0)
                f1_score = avg_metrics.get('avg_f1_score', 0)
                pr_auc = avg_metrics.get('avg_pr_auc', 0)
                
                f.write(f"{setting:<25} {accuracy:<15.4f} {precision:<15.4f} {recall:<15.4f} {f1_score:<15.4f} {pr_auc:<15.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"總結報告已保存到: {output_file}")

def main():
    """
    主函數
    """
    print("Lizz_data 評估結果讀取和轉換工具")
    print("="*50)
    
    # 檢查結果目錄
    results_dir = "lizz_evaluation_results"
    if not os.path.exists(results_dir):
        print(f"結果目錄不存在: {results_dir}")
        print("請先運行評估腳本生成結果")
        return
    
    # 處理所有結果
    process_all_results(results_dir)
    
    # 創建總結報告
    create_summary_report(results_dir)
    
    print("\n所有結果已轉換完成！")
    print("生成的文件:")
    print("- evaluation_results_csv/ - CSV格式結果")
    print("- evaluation_results_txt/ - TXT格式詳細報告")
    print("- summary_report.txt - 所有結果總結")

if __name__ == "__main__":
    main()
