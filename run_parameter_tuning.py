#!/usr/bin/env python3
"""
參數調優運行腳本
"""

import os
import sys

# 設定環境變數
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 導入並運行參數調優
from test_lizz_data_evaluation_unbalanced_tuned import parameter_tuning_experiment

if __name__ == "__main__":
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    random_seed = 42
    
    print("開始參數調優實驗...")
    print("這將測試多種參數組合來找到最佳設定")
    print("預計需要一些時間...")
    
    # 運行參數調優實驗
    results, best_accuracy, best_f1 = parameter_tuning_experiment(
        data_root, "00177", "5shots", 5, autoencoder_dir, random_seed
    )
    
    print("\n參數調優完成！")
    print("詳細結果已保存到 lizz_evaluation_results/ 目錄")
