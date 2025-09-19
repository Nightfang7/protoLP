#!/usr/bin/env python3
"""
運行Lizz_data評估 - Unbalanced版本
"""

from test_lizz_data_evaluation_unbalanced import (
    evaluate_lizz_35_cases_unbalanced, 
    batch_evaluate_lizz_data_unbalanced
)

def main():
    """
    主函數
    """
    print("Lizz_data Few-Shot Learning 評估 - Unbalanced版本")
    print("="*60)
    
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    output_dir = "lizz_evaluation_results"
    random_seed = 42
    
    print(f"資料根目錄: {data_root}")
    print(f"Autoencoder目錄: {autoencoder_dir}")
    print(f"結果輸出目錄: {output_dir}")
    print(f"隨機種子: {random_seed}")
    print()
    
    # 選擇評估模式
    print("請選擇評估模式:")
    print("1. 評估單個設定 (00177, 5shots, 5-way)")
    print("2. 批量評估所有設定")
    print("3. 自定義設定")
    
    choice = input("請輸入選擇 (1-3): ").strip()
    
    if choice == "1":
        # 評估單個設定
        print("\n開始評估 00177, 5shots, 5-way...")
        results = evaluate_lizz_35_cases_unbalanced(
            data_root, "00177", "5shots", 5, 
            autoencoder_dir, output_dir, random_seed
        )
        
    elif choice == "2":
        # 批量評估所有設定
        print("\n開始批量評估所有設定...")
        all_results = batch_evaluate_lizz_data_unbalanced(
            data_root, autoencoder_dir, output_dir, random_seed
        )
        
    elif choice == "3":
        # 自定義設定
        dataset = input("請輸入資料集名稱 (00177 或 apidms): ").strip()
        shot_type = input("請輸入shot類型 (5shots 或 10shots): ").strip()
        way = int(input("請輸入way數量 (5 或 10): "))
        
        print(f"\n開始評估 {dataset}, {shot_type}, {way}-way...")
        results = evaluate_lizz_35_cases_unbalanced(
            data_root, dataset, shot_type, way, 
            autoencoder_dir, output_dir, random_seed
        )
        
    else:
        print("無效選擇，使用預設設定...")
        results = evaluate_lizz_35_cases_unbalanced(
            data_root, "00177", "5shots", 5, 
            autoencoder_dir, output_dir, random_seed
        )
    
    print("\n評估完成！")
    print("結果已保存到 lizz_evaluation_results/ 目錄")

if __name__ == "__main__":
    main()
