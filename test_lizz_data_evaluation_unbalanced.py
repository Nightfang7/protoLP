#!/usr/bin/env python3
"""
Lizz_data Few-Shot Learning 評估腳本 - Unbalanced版本
使用unbalanced方法處理不同數量的query samples
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
        data = ndatas[i].cpu().numpy()  # 轉換為numpy
        
        # 檢查數據是否包含非有限值
        if not np.isfinite(data).all():
            print(f"警告: 數據包含非有限值，正在修復...")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"修復後數據範圍: [{data.min():.4f}, {data.max():.4f}]")
        
        # 使用PCA降維
        n_components = min(K, n_features, n_samples)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        
        # 檢查降維結果
        if reduced_data.shape[1] != K:
            print(f"警告: PCA降維後維度為 {reduced_data.shape[1]}，期望 {K}")
            # 如果維度不匹配，進行填充或截斷
            if reduced_data.shape[1] < K:
                # 填充零
                padding = np.zeros((reduced_data.shape[0], K - reduced_data.shape[1]))
                reduced_data = np.concatenate([reduced_data, padding], axis=1)
            else:
                # 截斷
                reduced_data = reduced_data[:, :K]
        
        # 轉換回torch tensor
        reduced_datas.append(torch.from_numpy(reduced_data).float())
    
    return torch.stack(reduced_datas, dim=0)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    計算準確率、precision、recall、F1-score和PR-AUC
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        y_prob: 預測機率
    
    Returns:
        dict: 包含所有指標的字典
    """
    accuracy = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 計算PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_prob, average='macro')
    except:
        pr_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pr_auc': pr_auc
    }

def evaluate_single_lizz_case_unbalanced(data_root, dataset_name, shot_type, way, case_id, 
                                       autoencoder_dir="autoencoder", verbose=False, random_seed=42):
    """
    使用unbalanced方法評估單個Lizz case
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        shot_type (str): shot類型 ('5shots' 或 '10shots')
        way (int): way數量 (5-10)
        case_id (int): case編號 (1-35)
        autoencoder_dir (str): autoencoder資料夾路徑
        verbose (bool): 是否顯示詳細信息
        random_seed (int): 隨機種子，預設為42
    
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
        
        # 重新標籤化support和query
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
        
        # 完整的數據預處理 - 恢復之前的設定
        print(f"使用完整預處理步驟")
        
        # 數據預處理 - 修復負值問題
        beta = 0.5
        # 確保所有值都是非負的，避免Power transformation產生NaN
        ndatas = torch.clamp(ndatas, min=0.0) + 1e-6
        ndatas = torch.pow(ndatas, beta)
        ndatas = scaleEachUnitaryDatas(ndatas)
        
        # 檢查數據是否包含非有限值
        if not torch.isfinite(ndatas).all():
            print(f"警告: 預處理前的數據包含非有限值，正在修復...")
            ndatas = torch.nan_to_num(ndatas, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用穩定的降維方法 - 優先使用PCA，避免SVD收斂問題
        try:
            print(f"使用PCA降維從 {ndatas.size(2)} 維降到 40 維...")
            ndatas = pca_reduction(ndatas, 40)
        except Exception as e:
            print(f"PCA降維失敗: {e}，嘗試SVD...")
            try:
                ndatas = SVDreduction(ndatas, 40)
            except Exception as e2:
                print(f"SVD降維也失敗: {e2}，保持原始維度...")
                # 如果都失敗，保持原始維度
        
        n_nfeat = ndatas.size(2)
        ndatas = centerDatas(ndatas)
        
        # 移到GPU
        ndatas = ndatas.cuda()
        labels = labels.cuda()
        
        # 設定參數 - unbalanced方法
        n_lsamples = n_shot * way  # support samples
        n_usamples = len(query_features)  # query samples (可以不同)
        
        # 創建模型
        lam = 10
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
                    
                    # 檢查特徵是否包含非有限值
                    if not torch.isfinite(class_features).all():
                        print(f"警告: 類別 {i} 的特徵包含非有限值，正在修復...")
                        class_features = torch.nan_to_num(class_features, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    prototype = class_features.mean(dim=0)
                    
                    # 檢查原型是否包含非有限值
                    if not torch.isfinite(prototype).all():
                        print(f"警告: 類別 {i} 的原型包含非有限值，正在修復...")
                        prototype = torch.nan_to_num(prototype, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    model.mus_ori[0, i] = prototype
                    model.mus[0, i] = prototype
        
        # 正規化原型
        model.mus = model.mus / model.mus.norm(dim=2, keepdim=True)
        
        # 簡化的預測方法 - 使用最近鄰分類
        # 計算query samples到每個原型的距離
        query_features_processed = ndatas[0, n_lsamples:]  # query features
        query_labels_true = labels[0]  # true labels (已經是query labels)
        
        # 檢查query特徵是否包含非有限值
        if not torch.isfinite(query_features_processed).all():
            print(f"警告: query特徵包含非有限值，正在修復...")
            query_features_processed = torch.nan_to_num(query_features_processed, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 檢查原型是否包含非有限值
        if not torch.isfinite(model.mus[0]).all():
            print(f"警告: 原型包含非有限值，正在修復...")
            model.mus[0] = torch.nan_to_num(model.mus[0], nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 計算距離
        distances = torch.cdist(query_features_processed, model.mus[0])
        
        # 檢查距離是否包含非有限值
        if not torch.isfinite(distances).all():
            print(f"警告: 距離矩陣包含非有限值，正在修復...")
            distances = torch.nan_to_num(distances, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
        
        predictions = distances.argmin(dim=1)
        
        # 計算指標
        y_true = query_labels_true.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # 調試信息
        if verbose or case_id <= 3:  # 顯示前3個cases的詳細信息
            print(f"\n=== Case {case_id} 調試信息 ===")
            print(f"Support set標籤分佈: {torch.bincount(support_labels_remapped)}")
            print(f"Query set標籤分佈: {torch.bincount(query_labels_remapped)}")
            print(f"預測標籤分佈: {np.bincount(y_pred)}")
            print(f"真實標籤分佈: {np.bincount(y_true)}")
            print(f"原型形狀: {model.mus[0].shape}")
            print(f"Query特徵形狀: {query_features_processed.shape}")
            print(f"距離矩陣形狀: {distances.shape}")
            print(f"距離範圍: [{distances.min():.4f}, {distances.max():.4f}]")
            
            # 檢查每個類別的原型是否正確計算
            for i in range(way):
                class_mask = (support_labels_remapped == i)
                if class_mask.any():
                    support_indices = torch.where(class_mask)[0]
                    print(f"類別 {i}: {len(support_indices)} 個support samples")
                    print(f"  原型範數: {model.mus[0, i].norm().item():.4f}")
                else:
                    print(f"類別 {i}: 沒有support samples!")
        
        # 計算預測機率 (基於距離的softmax)
        distances_neg = -distances  # 負距離用於softmax
        y_prob = torch.softmax(distances_neg, dim=1).cpu().numpy()
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # 計算準確率
        accuracy = (y_pred == y_true).mean()
        metrics['accuracy_mean'] = accuracy
        metrics['accuracy_std'] = 0.0  # 單次運行，標準差為0
        
        if verbose:
            print(f"Case {case_id} 結果:")
            print(f"  準確率: {accuracy:.4f}")
            print(f"  精確率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分數: {metrics['f1_score']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Case {case_id} 評估失敗: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'pr_auc': 0.0,
            'accuracy_mean': 0.0,
            'accuracy_std': 0.0
        }

def evaluate_lizz_35_cases_unbalanced(data_root, dataset_name, shot_type, way, 
                                     autoencoder_dir="autoencoder", output_dir="lizz_evaluation_results", 
                                     random_seed=42):
    """
    使用unbalanced方法評估35個Lizz cases
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        shot_type (str): shot類型 ('5shots' 或 '10shots')
        way (int): way數量 (5-10)
        autoencoder_dir (str): autoencoder資料夾路徑
        output_dir (str): 結果輸出目錄
        random_seed (int): 隨機種子，預設為42
    
    Returns:
        dict: 包含所有結果的字典
    """
    print(f"開始評估 {dataset_name} {shot_type} {way}-way (unbalanced方法)")
    print(f"使用隨機種子: {random_seed}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 評估35個cases
    all_metrics = []
    
    for case_id in tqdm(range(1, 36), desc="評估Cases"):
        metrics = evaluate_single_lizz_case_unbalanced(
            data_root, dataset_name, shot_type, way, case_id,
            autoencoder_dir, verbose=False, random_seed=random_seed
        )
        all_metrics.append(metrics)
    
    # 計算平均指標
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values)
    
    # 顯示結果
    print("\n" + "="*60)
    print(f"35個Cases的平均結果 ({dataset_name}, {shot_type}, {way}-way, unbalanced):")
    print("="*60)
    print(f"準確率 (Accuracy): {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"精確率 (Precision): {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"召回率 (Recall): {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    print(f"F1分數 (F1-Score): {avg_metrics['avg_f1_score']:.4f} ± {avg_metrics['std_f1_score']:.4f}")
    print(f"PR-AUC: {avg_metrics['avg_pr_auc']:.4f} ± {avg_metrics['std_pr_auc']:.4f}")
    print("="*60)
    
    # 保存詳細結果
    results = {
        'individual_cases': all_metrics,
        'average_metrics': avg_metrics,
        'config': {
            'dataset_name': dataset_name,
            'shot_type': shot_type,
            'way': way,
            'n_cases': 35,
            'method': 'unbalanced'
        }
    }
    
    results_path = os.path.join(
        output_dir, 
        f"lizz_evaluation_unbalanced_{dataset_name}_{shot_type}_{way}way.pkl"
    )
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"詳細結果已保存到: {results_path}")
    
    return results

def batch_evaluate_lizz_data_unbalanced(data_root, autoencoder_dir="autoencoder", 
                                       output_dir="lizz_evaluation_results", random_seed=42):
    """
    使用unbalanced方法批量評估所有Lizz_data設定
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        autoencoder_dir (str): autoencoder資料夾路徑
        output_dir (str): 結果輸出目錄
        random_seed (int): 隨機種子，預設為42
    """
    print("開始批量評估Lizz_data (unbalanced方法)")
    print("="*50)
    
    # 定義所有要評估的設定
    datasets = ['00177', 'apidms']
    shot_types = ['5shots', '10shots']
    ways = [5, 10]
    
    all_results = {}
    
    for dataset in datasets:
        for shot_type in shot_types:
            for way in ways:
                print(f"\n評估設定: {dataset}, {shot_type}, {way}-way")
                try:
                    results = evaluate_lizz_35_cases_unbalanced(
                        data_root, dataset, shot_type, way, 
                        autoencoder_dir, output_dir, random_seed
                    )
                    all_results[f"{dataset}_{shot_type}_{way}way"] = results
                except Exception as e:
                    print(f"評估 {dataset} {shot_type} {way}-way 失敗: {e}")
                    continue
    
    print("\n所有評估完成！")
    return all_results

if __name__ == "__main__":
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    output_dir = "lizz_evaluation_results"
    random_seed = 42
    
    # 評估單個設定
    results = evaluate_lizz_35_cases_unbalanced(
        data_root, "00177", "5shots", 5, 
        autoencoder_dir, output_dir, random_seed
    )
    
    print("\n評估完成！")
