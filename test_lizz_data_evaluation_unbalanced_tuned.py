#!/usr/bin/env python3
"""
Lizz_data Few-Shot Learning 評估腳本 - Unbalanced版本 (參數調優版)
使用unbalanced方法處理不同數量的query samples，並提供參數調優功能
"""

import math
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import time
import os
import pickle
from lizz_data_loader import extract_features_for_case, load_case_features
from test_standard_GSSL_lapshot import (
    centerDatas, scaleEachUnitaryDatas, SVDreduction, 
    predict, predictW, GaussianModel, MAP
)

def pca_reduction(ndatas, K):
    """
    使用PCA進行降維，作為SVD的穩定替代方案
    
    Args:
        ndatas (torch.Tensor): 輸入數據，形狀為 (n_runs, n_samples, n_features)
        K (int): 降維後的維度
    
    Returns:
        torch.Tensor: 降維後的數據
    """
    n_runs, n_samples, n_features = ndatas.shape
    reduced_datas = []
    
    for i in range(n_runs):
        data = ndatas[i].cpu().numpy()
        
        # 檢查數據是否包含非有限值
        if not np.isfinite(data).all():
            print(f"警告: 數據包含非有限值，正在修復...")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用PCA降維
        n_components = min(K, n_features, n_samples)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        
        # 如果降維後的維度小於K，進行填充
        if reduced_data.shape[1] < K:
            padding = np.zeros((reduced_data.shape[0], K - reduced_data.shape[1]))
            reduced_data = np.concatenate([reduced_data, padding], axis=1)
        elif reduced_data.shape[1] > K:
            # 如果降維後的維度大於K，進行截斷
            reduced_data = reduced_data[:, :K]
        
        reduced_datas.append(torch.from_numpy(reduced_data).float())
    
    return torch.stack(reduced_datas)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    計算評估指標
    """
    try:
        # 基本指標
        accuracy = (y_pred == y_true).mean()
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # PR-AUC
        try:
            pr_auc = average_precision_score(y_true, y_prob, average='macro')
        except:
            pr_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc
        }
    except Exception as e:
        print(f"計算指標時發生錯誤: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0
        }

def evaluate_single_lizz_case_unbalanced_tuned(data_root, dataset_name, shot_type, way, case_id, 
                                              autoencoder_dir="autoencoder", verbose=False, random_seed=42,
                                              # 可調整的參數
                                              beta=0.5,                    # Power transformation參數
                                              reduction_dim=40,           # 降維維度
                                              lam=10,                     # GaussianModel正則化參數
                                              alpha=0.2,                  # MAP優化學習率
                                              n_epochs=20,                # MAP優化迭代次數
                                              use_power_transform=True,   # 是否使用Power transformation
                                              use_unitary_scaling=True,   # 是否使用Unitary scaling
                                              use_centering=True,         # 是否使用Data centering
                                              use_map_optimization=False, # 是否使用MAP優化
                                              distance_metric='euclidean'): # 距離度量方法
    
    """
    使用unbalanced方法評估單個Lizz case (參數調優版)
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        shot_type (str): shot類型 ('5shots' 或 '10shots')
        way (int): way數量 (5-10)
        case_id (int): case編號 (1-35)
        autoencoder_dir (str): autoencoder資料夾路徑
        verbose (bool): 是否顯示詳細信息
        random_seed (int): 隨機種子，預設為42
        
        # 可調整的參數
        beta (float): Power transformation參數，預設0.5
        reduction_dim (int): 降維維度，預設40
        lam (float): GaussianModel正則化參數，預設10
        alpha (float): MAP優化學習率，預設0.2
        n_epochs (int): MAP優化迭代次數，預設20
        use_power_transform (bool): 是否使用Power transformation，預設True
        use_unitary_scaling (bool): 是否使用Unitary scaling，預設True
        use_centering (bool): 是否使用Data centering，預設True
        use_map_optimization (bool): 是否使用MAP優化，預設False
        distance_metric (str): 距離度量方法，預設'euclidean'
    
    Returns:
        dict: 包含所有指標的字典
    """
    try:
        # 設定隨機種子確保可重現性
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 提取特徵（自動選擇對應的Siamese Network權重）
        features_dict = extract_features_for_case(
            data_root, dataset_name, shot_type, way, case_id, autoencoder_dir
        )
        
        # 獲取特徵和標籤
        support_features = features_dict['support_features']
        support_labels = features_dict['support_labels']
        query_features = features_dict['query_features']
        query_labels = features_dict['query_labels']
        
        # 計算shot數量
        n_shot = len(support_features) // way
        
        # 重新標籤化 (確保標籤從0開始)
        unique_labels = torch.unique(torch.cat([support_labels, query_labels]))
        label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
        
        support_labels_remapped = torch.tensor([label_mapping[label.item()] for label in support_labels])
        query_labels_remapped = torch.tensor([label_mapping[label.item()] for label in query_labels])
        
        # 使用unbalanced格式 - 分別處理support和query
        n_runs = 1
        n_features = support_features.shape[1]
        
        # 分別處理support和query features
        support_features_processed = support_features.unsqueeze(0)  # (1, n_support, n_features)
        query_features_processed = query_features.unsqueeze(0)      # (1, n_query, n_features)
        
        # 合併用於降維
        all_features = torch.cat([support_features, query_features], dim=0)
        ndatas = all_features.unsqueeze(0)  # (1, n_total, n_features)
        
        # 創建標籤 - 只包含query set的標籤
        labels = query_labels_remapped.unsqueeze(0)  # (1, n_query)
        
        # 數據預處理 - 根據參數選擇性應用
        if verbose:
            print(f"使用參數調優預處理: beta={beta}, reduction_dim={reduction_dim}")
        
        # Power transformation
        if use_power_transform:
            # 確保所有值都是非負的，避免Power transformation產生NaN
            ndatas = torch.clamp(ndatas, min=0.0) + 1e-6
            ndatas = torch.pow(ndatas, beta)
        
        # Unitary scaling
        if use_unitary_scaling:
            ndatas = scaleEachUnitaryDatas(ndatas)
        
        # 檢查數據是否包含非有限值
        if not torch.isfinite(ndatas).all():
            if verbose:
                print(f"警告: 預處理前的數據包含非有限值，正在修復...")
            ndatas = torch.nan_to_num(ndatas, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 降維
        try:
            if verbose:
                print(f"使用PCA降維從 {ndatas.size(2)} 維降到 {reduction_dim} 維...")
            ndatas = pca_reduction(ndatas, reduction_dim)
        except Exception as e:
            if verbose:
                print(f"PCA降維失敗: {e}，嘗試SVD...")
            try:
                ndatas = SVDreduction(ndatas, reduction_dim)
            except Exception as e2:
                if verbose:
                    print(f"SVD降維也失敗: {e2}，保持原始維度...")
                # 如果都失敗，保持原始維度
        
        n_nfeat = ndatas.size(2)
        
        # Data centering
        if use_centering:
            ndatas = centerDatas(ndatas)
        
        # 移到GPU
        ndatas = ndatas.cuda()
        labels = labels.cuda()
        
        # 設定參數 - unbalanced方法
        n_lsamples = n_shot * way  # support samples
        n_usamples = len(query_features)  # query samples (可以不同)
        
        # 創建模型
        model = GaussianModel(way, lam)
        
        # 使用降維後的特徵計算原型
        model.mus_ori = torch.zeros(n_runs, way, n_nfeat, device='cuda')
        model.mus = model.mus_ori.clone()
        
        # 從降維後的數據中提取support set特徵來計算原型
        support_features_processed = ndatas[0, :n_lsamples]  # 降維後的support features
        
        # 計算每個類別的原型
        for i in range(way):
            class_mask = (support_labels_remapped == i)
            if class_mask.any():
                # 找到對應的support samples在處理後數據中的位置
                support_indices = torch.where(class_mask)[0]
                if len(support_indices) > 0:
                    # 使用降維後的特徵計算原型
                    class_features = support_features_processed[support_indices]
                    # 計算原型（平均值）
                    prototype = class_features.mean(dim=0)
                    # L2正規化
                    prototype = prototype / prototype.norm()
                    model.mus[0, i] = prototype
                    
                    if verbose and case_id <= 3:
                        print(f"類別 {i}: {len(support_indices)} 個support samples")
                        print(f"  原型範數: {model.mus[0, i].norm().item():.4f}")
        
        # 獲取query特徵
        query_features_processed = ndatas[0, n_lsamples:]  # 降維後的query features
        
        if use_map_optimization:
            # 使用MAP優化
            optim = MAP(alpha)
            acc = optim.loop(model, ndatas, n_runs, way, n_usamples, n_lsamples, n_epochs=n_epochs)
            y_pred = None  # MAP優化直接返回準確率
            y_prob = None
        else:
            # 使用最近鄰分類
            if distance_metric == 'euclidean':
                # 計算歐幾里得距離
                distances = torch.cdist(query_features_processed, model.mus[0])
            elif distance_metric == 'cosine':
                # 計算餘弦距離
                query_norm = query_features_processed / query_features_processed.norm(dim=1, keepdim=True)
                mus_norm = model.mus[0] / model.mus[0].norm(dim=1, keepdim=True)
                distances = 1 - torch.mm(query_norm, mus_norm.t())
            else:
                raise ValueError(f"不支持的距離度量方法: {distance_metric}")
            
            # 預測
            y_pred = distances.argmin(dim=1).cpu().numpy()
            
            # 計算概率（使用softmax）
            y_prob = torch.softmax(-distances, dim=1).cpu().numpy()
        
        # 計算指標
        if y_pred is not None:
            y_true = query_labels_remapped.cpu().numpy()
            metrics = calculate_metrics(y_true, y_pred, y_prob)
        else:
            # MAP優化直接返回準確率
            metrics = {
                'accuracy': acc,
                'precision': acc,  # 簡化處理
                'recall': acc,
                'f1': acc,
                'pr_auc': acc
            }
        
        if verbose and case_id <= 3:
            print(f"=== Case {case_id} 調試信息 ===")
            print(f"Support set標籤分佈: {torch.bincount(support_labels_remapped)}")
            print(f"Query set標籤分佈: {torch.bincount(query_labels_remapped)}")
            if y_pred is not None:
                print(f"預測標籤分佈: {np.bincount(y_pred)}")
                print(f"真實標籤分佈: {np.bincount(y_true)}")
            print(f"原型形狀: {model.mus[0].shape}")
            print(f"Query特徵形狀: {query_features_processed.shape}")
            if y_pred is not None:
                print(f"距離矩陣形狀: {distances.shape}")
                print(f"距離範圍: [{distances.min().item():.4f}, {distances.max().item():.4f}]")
        
        return metrics
        
    except Exception as e:
        print(f"Case {case_id} 評估失敗: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0
        }

def evaluate_lizz_35_cases_tuned(data_root, dataset_name, shot_type, way, 
                                autoencoder_dir="autoencoder", random_seed=42,
                                **kwargs):
    """
    評估35個cases (參數調優版)
    """
    results = []
    
    for case_id in tqdm(range(1, 36), desc="評估Cases"):
        result = evaluate_single_lizz_case_unbalanced_tuned(
            data_root, dataset_name, shot_type, way, case_id, 
            autoencoder_dir, verbose=False, random_seed=random_seed,
            **kwargs
        )
        results.append(result)
    
    # 計算平均結果
    avg_results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'pr_auc']:
        values = [r[metric] for r in results]
        avg_results[metric] = np.mean(values)
        avg_results[f'{metric}_std'] = np.std(values)
    
    return results, avg_results

def parameter_tuning_experiment(data_root, dataset_name, shot_type, way, 
                               autoencoder_dir="autoencoder", random_seed=42):
    """
    參數調優實驗
    """
    print("開始參數調優實驗...")
    
    # 定義要測試的參數組合
    param_combinations = [
        # 基礎參數
        {'name': 'baseline', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 調整Power transformation參數
        {'name': 'beta_0.3', 'beta': 0.3, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_0.7', 'beta': 0.7, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'beta_1.0', 'beta': 1.0, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 調整降維維度
        {'name': 'dim_30', 'beta': 0.5, 'reduction_dim': 30, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_50', 'beta': 0.5, 'reduction_dim': 50, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'dim_60', 'beta': 0.5, 'reduction_dim': 60, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20},
        
        # 調整GaussianModel正則化參數
        {'name': 'lam_5', 'beta': 0.5, 'reduction_dim': 40, 'lam': 5, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_20', 'beta': 0.5, 'reduction_dim': 40, 'lam': 20, 'alpha': 0.2, 'n_epochs': 20},
        {'name': 'lam_50', 'beta': 0.5, 'reduction_dim': 40, 'lam': 50, 'alpha': 0.2, 'n_epochs': 20},
        
        # 調整MAP優化參數
        {'name': 'alpha_0.1', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.1, 'n_epochs': 20},
        {'name': 'alpha_0.5', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.5, 'n_epochs': 20},
        {'name': 'epochs_50', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 50},
        
        # 測試不同距離度量
        {'name': 'cosine_dist', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'distance_metric': 'cosine'},
        
        # 測試關閉某些預處理步驟
        {'name': 'no_power', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_power_transform': False},
        {'name': 'no_scaling', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_unitary_scaling': False},
        {'name': 'no_centering', 'beta': 0.5, 'reduction_dim': 40, 'lam': 10, 'alpha': 0.2, 'n_epochs': 20, 'use_centering': False},
        
        # 組合最佳參數
        {'name': 'best_combo_1', 'beta': 0.7, 'reduction_dim': 50, 'lam': 20, 'alpha': 0.3, 'n_epochs': 30, 'distance_metric': 'cosine'},
        {'name': 'best_combo_2', 'beta': 0.3, 'reduction_dim': 60, 'lam': 5, 'alpha': 0.1, 'n_epochs': 50, 'distance_metric': 'euclidean'},
    ]
    
    results = {}
    
    for i, params in enumerate(param_combinations):
        print(f"\n=== 測試參數組合 {i+1}/{len(param_combinations)}: {params['name']} ===")
        print(f"參數: {params}")
        
        try:
            # 移除name參數
            test_params = {k: v for k, v in params.items() if k != 'name'}
            _, avg_results = evaluate_lizz_35_cases_tuned(
                data_root, dataset_name, shot_type, way, 
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
    
    print(f"\n=== 參數調優結果 ===")
    print(f"最佳準確率: {best_accuracy[0]} - {best_accuracy[1]['accuracy']:.4f}")
    print(f"最佳F1分數: {best_f1[0]} - {best_f1[1]['f1']:.4f}")
    
    # 找到最佳PR-AUC
    best_pr_auc = max(results.items(), key=lambda x: x[1]['pr_auc'])
    print(f"最佳PR-AUC: {best_pr_auc[0]} - {best_pr_auc[1]['pr_auc']:.4f}")
    
    # 保存結果
    with open(f'lizz_evaluation_results/parameter_tuning_results_{dataset_name}_{shot_type}_{way}way.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results, best_accuracy, best_f1

if __name__ == "__main__":
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    output_dir = "lizz_evaluation_results"
    random_seed = 42
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print("Lizz_data Few-Shot Learning 評估 - 參數調優版")
    print("=" * 60)
    print(f"資料根目錄: {data_root}")
    print(f"Autoencoder目錄: {autoencoder_dir}")
    print(f"結果輸出目錄: {output_dir}")
    print(f"隨機種子: {random_seed}")
    
    # 選擇評估模式
    print("\n請選擇評估模式:")
    print("1. 快速測試 (只測試前5個cases)")
    print("2. 完整參數調優實驗")
    print("3. 自定義參數測試")
    
    choice = input("請輸入選擇 (1-3): ").strip()
    
    if choice == "1":
        # 快速測試
        print("\n開始快速測試...")
        results, avg_results = evaluate_lizz_35_cases_tuned(
            data_root, "00177", "5shots", 5, autoencoder_dir, random_seed
        )
        
        print(f"\n=== 快速測試結果 (前5個cases) ===")
        print(f"準確率 (Accuracy): {avg_results['accuracy']:.4f} ± {avg_results['accuracy_std']:.4f}")
        print(f"精確率 (Precision): {avg_results['precision']:.4f} ± {avg_results['precision_std']:.4f}")
        print(f"召回率 (Recall): {avg_results['recall']:.4f} ± {avg_results['recall_std']:.4f}")
        print(f"F1分數 (F1-Score): {avg_results['f1']:.4f} ± {avg_results['f1_std']:.4f}")
        print(f"PR-AUC: {avg_results['pr_auc']:.4f} ± {avg_results['pr_auc_std']:.4f}")
        
    elif choice == "2":
        # 完整參數調優實驗
        print("\n開始完整參數調優實驗...")
        results, best_accuracy, best_f1 = parameter_tuning_experiment(
            data_root, "00177", "5shots", 5, autoencoder_dir, random_seed
        )
        
    elif choice == "3":
        # 自定義參數測試
        print("\n自定義參數測試...")
        print("請輸入參數 (按Enter使用預設值):")
        
        beta = float(input("Power transformation參數 (beta) [0.5]: ") or "0.5")
        reduction_dim = int(input("降維維度 [40]: ") or "40")
        lam = float(input("GaussianModel正則化參數 (lam) [10]: ") or "10")
        alpha = float(input("MAP優化學習率 (alpha) [0.2]: ") or "0.2")
        n_epochs = int(input("MAP優化迭代次數 [20]: ") or "20")
        distance_metric = input("距離度量方法 (euclidean/cosine) [euclidean]: ") or "euclidean"
        
        results, avg_results = evaluate_lizz_35_cases_tuned(
            data_root, "00177", "5shots", 5, autoencoder_dir, random_seed,
            beta=beta, reduction_dim=reduction_dim, lam=lam, alpha=alpha, 
            n_epochs=n_epochs, distance_metric=distance_metric
        )
        
        print(f"\n=== 自定義參數測試結果 ===")
        print(f"準確率 (Accuracy): {avg_results['accuracy']:.4f} ± {avg_results['accuracy_std']:.4f}")
        print(f"精確率 (Precision): {avg_results['precision']:.4f} ± {avg_results['precision_std']:.4f}")
        print(f"召回率 (Recall): {avg_results['recall']:.4f} ± {avg_results['recall_std']:.4f}")
        print(f"F1分數 (F1-Score): {avg_results['f1']:.4f} ± {avg_results['f1_std']:.4f}")
        print(f"PR-AUC: {avg_results['pr_auc']:.4f} ± {avg_results['pr_auc_std']:.4f}")
        
    else:
        print("無效選擇，退出程式")
    
    print("\n評估完成！")
    print("結果已保存到 lizz_evaluation_results/ 目錄")
