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
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用PCA降維
        pca = PCA(n_components=min(K, n_features, n_samples-1))
        reduced_data = pca.fit_transform(data)
        
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
    # 基本指標
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # PR-AUC計算
    try:
        # 將y_prob轉換為numpy array並確保是2D
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        if y_prob.ndim == 3:  # 如果是3D (n_runs, n_samples, n_classes)
            y_prob = y_prob.reshape(-1, y_prob.shape[-1])
            y_true_flat = y_true.reshape(-1)
        else:
            y_true_flat = y_true
        
        # 計算PR-AUC
        pr_auc = average_precision_score(y_true_flat, y_prob, average='macro')
    except Exception as e:
        print(f"PR-AUC計算錯誤: {e}")
        pr_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pr_auc': pr_auc
    }

def evaluate_single_lizz_case(data_root, dataset_name, shot_type, way, case_id, 
                             autoencoder_dir="autoencoder", verbose=False, random_seed=42):
    """
    評估單個Lizz case
    
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
        # 提取特徵
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
        total_queries = len(query_features)
        
        # 計算每個類別的query數量（標準方法要求每個類別有相同數量的query）
        n_queries = total_queries // way
        
        # 重新組織數據以符合FSLTask格式
        # 按照原始FSLTask的格式組織: (n_runs, n_shot+n_queries, n_ways, n_features)
        
        # 重新標籤化 (確保標籤從0開始)
        unique_labels = torch.unique(torch.cat([support_labels, query_labels]))
        label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
        
        # 重新標籤化support和query
        support_labels_remapped = torch.tensor([label_mapping[label.item()] for label in support_labels])
        query_labels_remapped = torch.tensor([label_mapping[label.item()] for label in query_labels])
        
        # 計算每個類別的樣本數量
        n_shot = len(support_features) // way
        # n_queries 已經在上面定義了，不需要重複定義
        
        # 重新組織support set: (n_ways, n_shot, n_features)
        support_organized = []
        for i in range(way):
            class_mask = (support_labels_remapped == i)
            class_features = support_features[class_mask]
            support_organized.append(class_features)
        
        # 重新組織query set: 確保每個類別有相同數量的query samples
        query_organized = []
        min_queries = float('inf')
        
        # 先找到每個類別的最小query數量
        for i in range(way):
            class_mask = (query_labels_remapped == i)
            class_features = query_features[class_mask]
            min_queries = min(min_queries, len(class_features))
        
        # 確保每個類別有相同數量的query samples
        for i in range(way):
            class_mask = (query_labels_remapped == i)
            class_features = query_features[class_mask]
            if len(class_features) > min_queries:
                # 如果query samples太多，隨機選擇
                indices = torch.randperm(len(class_features))[:min_queries]
                class_features = class_features[indices]
            query_organized.append(class_features)
        
        # 合併support和query: (n_ways, n_shot+n_queries, n_features)
        all_organized = []
        for i in range(way):
            class_data = torch.cat([support_organized[i], query_organized[i]], dim=0)
            all_organized.append(class_data)
        
        # 設定n_queries為每個類別的query數量
        n_queries = min_queries
        
        # 轉換為FSLTask格式: (n_runs, n_shot+n_queries, n_ways, n_features)
        n_runs = 1
        n_features = support_features.shape[1]
        
        # 重新組織數據
        ndatas = torch.stack(all_organized, dim=1).unsqueeze(0)  # (1, n_shot+n_queries, n_ways, n_features)
        
        # 創建標籤
        labels = torch.arange(way).unsqueeze(0).repeat(1, n_shot + n_queries)  # (1, n_shot+n_queries)
        
        # 數據預處理 - 先重新整形為降維函數期望的格式
        # 將 (n_runs, n_shot+n_queries, n_ways, n_features) 轉換為 (n_runs, n_samples, n_features)
        n_samples = ndatas.size(1) * ndatas.size(2)  # n_shot+n_queries * n_ways
        ndatas_flat = ndatas.view(n_runs, n_samples, n_features)
        
        # 數據預處理
        beta = 0.5
        ndatas_flat = torch.pow(ndatas_flat + 1e-6, beta)
        ndatas_flat = scaleEachUnitaryDatas(ndatas_flat)
        
        # 檢查數據是否包含非有限值
        if not torch.isfinite(ndatas_flat).all():
            print(f"警告: 預處理前的數據包含非有限值，正在修復...")
            ndatas_flat = torch.nan_to_num(ndatas_flat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用穩定的降維方法 - 優先使用PCA，避免SVD收斂問題
        try:
            print(f"使用PCA降維從 {ndatas_flat.size(2)} 維降到 40 維...")
            ndatas_flat = pca_reduction(ndatas_flat, 40)
        except Exception as e:
            print(f"PCA降維失敗: {e}，嘗試SVD...")
            try:
                ndatas_flat = SVDreduction(ndatas_flat, 40)
            except Exception as e2:
                print(f"SVD降維也失敗: {e2}，保持原始維度...")
                # 如果都失敗，保持原始維度
        
        n_nfeat = ndatas_flat.size(2)
        ndatas_flat = centerDatas(ndatas_flat)
        
        # 最終檢查
        if not torch.isfinite(ndatas_flat).all():
            print(f"警告: 預處理後的數據包含非有限值，正在修復...")
            ndatas_flat = torch.nan_to_num(ndatas_flat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 重新整形回FSLTask格式: (n_runs, n_shot+n_queries, n_ways, n_features)
        ndatas = ndatas_flat.view(n_runs, n_shot + n_queries, way, n_nfeat)
        
        # 移到GPU
        ndatas = ndatas.cuda()
        labels = labels.cuda()
        
        # 設定參數
        n_lsamples = n_shot * way
        n_usamples = n_queries * way
        
        # 創建模型
        lam = 10
        model = GaussianModel(way, lam)
        model.initFromLabelledDatas(ndatas, n_runs, n_shot, n_queries, way, n_nfeat)
        
        # 創建優化器
        alpha = 0.2
        optim = MAP(alpha)
        optim.verbose = verbose
        optim.progressBar = False
        
        # 訓練模型
        acc_test = optim.loop(model, ndatas, n_runs, way, n_usamples, n_lsamples, n_epochs=50)
        
        # 獲取最終預測
        final_probas = model.getProbas(ndatas, n_runs, way, n_usamples, n_lsamples)
        
        # 計算指標
        # 重新組織query set的真實標籤和預測結果
        y_true = []
        y_prob = []
        
        for i in range(way):
            # 每個類別的真實標籤
            class_labels = torch.full((n_queries,), i, dtype=torch.long)
            y_true.append(class_labels)
            
            # 每個類別的預測機率
            class_probas = final_probas[0, n_lsamples:, i]  # 對應類別的機率
            y_prob.append(class_probas)
        
        y_true = torch.cat(y_true).cpu().numpy()
        y_prob = torch.cat(y_prob, dim=0).cpu().numpy()
        y_pred = y_prob.argmax(axis=1)
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['accuracy_mean'] = acc_test[0]
        metrics['accuracy_std'] = acc_test[1]
        
        if verbose:
            print(f"Case {case_id}: Accuracy={metrics['accuracy']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}, "
                  f"PR-AUC={metrics['pr_auc']:.4f}")
        
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

def evaluate_lizz_35_cases(data_root, dataset_name, shot_type, way, 
                          autoencoder_dir="autoencoder", output_dir="lizz_evaluation_results", random_seed=42):
    """
    評估Lizz_data的35個cases並計算平均指標
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        shot_type (str): shot類型 ('5shots' 或 '10shots')
        way (int): way數量 (5-10)
        autoencoder_dir (str): autoencoder資料夾路徑
        output_dir (str): 輸出目錄
        random_seed (int): 隨機種子，預設為42
    
    Returns:
        dict: 包含所有平均指標的字典
    """
    print(f"開始評估Lizz_data: {dataset_name}, {shot_type}, {way}-way")
    print("="*60)
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 評估35個cases
    all_metrics = []
    
    for case_id in tqdm(range(1, 36), desc="評估Cases"):
        metrics = evaluate_single_lizz_case(
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
    print(f"35個Cases的平均結果 ({dataset_name}, {shot_type}, {way}-way):")
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
            'n_cases': 35
        }
    }
    
    results_path = os.path.join(
        output_dir, 
        f"lizz_evaluation_{dataset_name}_{shot_type}_{way}way.pkl"
    )
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"詳細結果已保存到: {results_path}")
    
    return avg_metrics

def batch_evaluate_lizz_data(data_root, autoencoder_dir="autoencoder", output_dir="lizz_evaluation_results", random_seed=42):
    """
    批量評估所有Lizz_data設定
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        autoencoder_dir (str): autoencoder資料夾路徑
        output_dir (str): 輸出目錄
        random_seed (int): 隨機種子，預設為42
    """
    # 定義所有要評估的設定
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
    
    all_results = {}
    
    for dataset_name, shot_type, way in configs:
        print(f"\n評估設定: {dataset_name}, {shot_type}, {way}-way")
        print("="*60)
        
        try:
            results = evaluate_lizz_35_cases(
                data_root, dataset_name, shot_type, way, 
                autoencoder_dir, output_dir, random_seed
            )
            all_results[f"{dataset_name}_{shot_type}_{way}way"] = results
            
        except Exception as e:
            print(f"設定 {dataset_name}_{shot_type}_{way}way 評估失敗: {e}")
            continue
    
    # 保存所有結果
    all_results_path = os.path.join(output_dir, "all_lizz_evaluation_results.pkl")
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n所有評估完成！結果保存在: {all_results_path}")
    
    # 顯示總結
    print("\n" + "="*80)
    print("評估總結:")
    print("="*80)
    for config_name, results in all_results.items():
        print(f"{config_name}:")
        print(f"  準確率: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"  F1分數: {results['avg_f1_score']:.4f} ± {results['std_f1_score']:.4f}")
        print(f"  PR-AUC: {results['avg_pr_auc']:.4f} ± {results['std_pr_auc']:.4f}")
        print()

def main():
    """
    主函數
    """
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"
    output_dir = "lizz_evaluation_results"
    
    # 檢查路徑
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
    except Exception as e:
        print(f"錯誤: {e}")
        return
    
    print("Lizz_data Few-Shot Learning 評估工具")
    print("="*50)
    print("可用的資料集: 00177, apidms")
    print("可用的shot類型: 5shots, 10shots")
    print("可用的way數量: 5, 6, 7, 8, 9, 10")
    print(f"隨機種子設定: 42 (確保結果可重現)")
    print()
    
    choice = input("選擇評估模式:\n1. 評估所有設定\n2. 評估特定設定\n請選擇 (1-2): ").strip()
    
    if choice == "1":
        # 評估所有設定
        batch_evaluate_lizz_data(data_root, autoencoder_dir, output_dir)
        
    elif choice == "2":
        # 評估特定設定
        dataset_name = input("請輸入資料集名稱 (00177 或 apidms): ").strip()
        shot_type = input("請輸入shot類型 (5shots 或 10shots): ").strip()
        way = int(input("請輸入way數量 (5-10): ").strip())
        
        results = evaluate_lizz_35_cases(
            data_root, dataset_name, shot_type, way, 
            autoencoder_dir, output_dir
        )
        
    else:
        print("無效的選擇")

if __name__ == "__main__":
    main()
