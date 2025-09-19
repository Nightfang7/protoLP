#!/usr/bin/env python3
"""
Lizz_data評估腳本
使用此腳本來評估您的autoencoder.pth檔案在Lizz_data上的表現
"""

import os
import sys
from test_lizz_data_evaluation import evaluate_lizz_35_cases, batch_evaluate_lizz_data

def main():
    print("Lizz_data Few-Shot Learning 評估工具")
    print("="*50)
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    
    # 檢查資料夾是否存在
    if not os.path.exists(data_root):
        print(f"錯誤: Lizz_data資料夾不存在: {data_root}")
        return
    
    if not os.path.exists(autoencoder_dir):
        print(f"錯誤: autoencoder資料夾不存在: {autoencoder_dir}")
        return
    
    # 檢查autoencoder檔案
    from lizz_data_loader import get_autoencoder_path
    try:
        get_autoencoder_path("00177", autoencoder_dir)
        get_autoencoder_path("apidms", autoencoder_dir)
        print("✓ 找到所有必要的autoencoder檔案")
        print("  - 00177 資料集使用: split3_42_autoencoder_model.pth")
        print("  - apidms 資料集使用: apidms_42_autoencoder_model.pth")
    except Exception as e:
        print(f"錯誤: {e}")
        return
    
    print("\n可用的資料集和設定:")
    print("資料集: 00177, apidms")
    print("Shot類型: 5shots, 10shots")
    print("Way數量: 5, 6, 7, 8, 9, 10")
    print("每個設定包含35個cases")
    print("隨機種子: 42 (確保結果可重現)")
    
    # 獲取評估模式
    print("\n請選擇評估模式:")
    print("1. 評估所有設定 (8個設定)")
    print("2. 評估特定設定")
    
    choice = input("請選擇 (1-2): ").strip()
    
    if choice == "1":
        # 評估所有設定
        print("\n將評估以下8個設定:")
        configs = [
            ("00177", "5shots", 5),
            ("00177", "5shots", 10),
            ("00177", "10shots", 5),
            ("00177", "10shots", 10),
            ("apidms", "5shots", 5),
            ("apidms", "5shots", 10),
            ("apidms", "10shots", 5),
            ("apidms", "10shots", 10),
        ]
        
        for i, (dataset, shot, way) in enumerate(configs, 1):
            print(f"{i}. {dataset} {shot} {way}-way")
        
        confirm = input("\n是否繼續? (y/n): ").strip().lower()
        if confirm != 'y':
            print("評估已取消")
            return
        
        try:
            batch_evaluate_lizz_data(data_root, autoencoder_dir)
            print("\n所有評估完成！")
            
        except Exception as e:
            print(f"評估過程中發生錯誤: {e}")
    
    elif choice == "2":
        # 評估特定設定
        print("\n請選擇要評估的設定:")
        
        # 選擇資料集
        print("1. 00177")
        print("2. apidms")
        dataset_choice = input("請選擇資料集 (1-2): ").strip()
        dataset_name = "00177" if dataset_choice == "1" else "apidms"
        
        # 選擇shot類型
        print("\n1. 5shots")
        print("2. 10shots")
        shot_choice = input("請選擇shot類型 (1-2): ").strip()
        shot_type = "5shots" if shot_choice == "1" else "10shots"
        
        # 選擇way數量
        way = int(input("請輸入way數量 (5-10): ").strip())
        
        print(f"\n將評估: {dataset_name} {shot_type} {way}-way")
        print("這將評估35個cases並計算平均指標")
        
        confirm = input("是否繼續? (y/n): ").strip().lower()
        if confirm != 'y':
            print("評估已取消")
            return
        
        try:
            results = evaluate_lizz_35_cases(
                data_root, dataset_name, shot_type, way, autoencoder_dir
            )
            print("\n評估完成！")
            
        except Exception as e:
            print(f"評估過程中發生錯誤: {e}")
    
    else:
        print("無效的選擇")

if __name__ == "__main__":
    main()
