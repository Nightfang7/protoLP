import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from autoencoder_feature_extractor import AutoencoderFeatureExtractor

class LizzDataset(Dataset):
    """
    Lizz_data資料集載入器
    支援按case載入support set和query set
    """
    
    def __init__(self, data_root, dataset_name, shot_type, way, case_id, 
                 set_type='known', transform=None):
        """
        初始化資料集
        
        Args:
            data_root (str): Lizz_data資料夾路徑
            dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
            shot_type (str): shot類型 ('5shots' 或 '10shots')
            way (int): way數量 (5-10)
            case_id (int): case編號 (1-35)
            set_type (str): 資料集類型 ('known' 或 'unknown')
            transform: 圖片轉換
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.shot_type = shot_type
        self.way = way
        self.case_id = case_id
        self.set_type = set_type
        self.transform = transform
        
        # 構建路徑
        self.base_path = os.path.join(
            data_root, 
            f"{dataset_name}_{shot_type}_5_10_35cases"
        )
        
        if set_type == 'known':
            self.case_path = os.path.join(
                self.base_path, 
                "known_with_case", 
                f"test_{way}_{case_id}"
            )
        else:  # unknown
            self.case_path = os.path.join(
                self.base_path, 
                "unknown_with_case", 
                f"test_{way}_{case_id}"
            )
        
        # 載入類別和圖片路徑
        self.classes, self.samples = self._load_samples()
        
    def _load_samples(self):
        """載入樣本和類別"""
        classes = []
        samples = []
        
        if not os.path.exists(self.case_path):
            raise ValueError(f"Case路徑不存在: {self.case_path}")
        
        # 獲取所有類別資料夾
        class_dirs = [d for d in os.listdir(self.case_path) 
                     if os.path.isdir(os.path.join(self.case_path, d))]
        class_dirs.sort()  # 確保順序一致
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.case_path, class_name)
            
            # 獲取該類別的所有圖片
            if self.set_type == 'known':
                # known_with_case有雙層資料夾結構
                inner_class_path = os.path.join(class_path, class_name)
                if os.path.exists(inner_class_path):
                    class_path = inner_class_path
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                samples.append((img_path, class_idx))
            
            classes.append(class_name)
        
        return classes, samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 載入圖片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """獲取類別名稱"""
        return self.classes

def create_lizz_data_loader(data_root, dataset_name, shot_type, way, case_id, 
                           set_type='known', batch_size=32, transform=None):
    """
    創建Lizz資料的DataLoader
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱
        shot_type (str): shot類型
        way (int): way數量
        case_id (int): case編號
        set_type (str): 資料集類型
        batch_size (int): 批次大小
        transform: 圖片轉換
    
    Returns:
        DataLoader: 資料載入器
    """
    dataset = LizzDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        shot_type=shot_type,
        way=way,
        case_id=case_id,
        set_type=set_type,
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 保持順序
        num_workers=0
    )

def get_autoencoder_path(dataset_name, autoencoder_dir="autoencoder"):
    """
    根據資料集名稱獲取對應的autoencoder檔案路徑
    
    Args:
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        autoencoder_dir (str): autoencoder資料夾路徑
    
    Returns:
        str: autoencoder檔案路徑
    """
    if dataset_name == "00177":
        autoencoder_file = "split3_42_autoencoder_model.pth"
    elif dataset_name == "apidms":
        autoencoder_file = "apidms_42_autoencoder_model.pth"
    else:
        raise ValueError(f"不支援的資料集名稱: {dataset_name}")
    
    autoencoder_path = os.path.join(autoencoder_dir, autoencoder_file)
    
    if not os.path.exists(autoencoder_path):
        raise FileNotFoundError(f"找不到對應的autoencoder檔案: {autoencoder_path}")
    
    return autoencoder_path

def get_siamese_weights_path(dataset_name, shot_type, model_dir="model"):
    """
    根據資料集名稱和shot類型獲取對應的Siamese Network權重檔案路徑
    
    Args:
        dataset_name (str): 資料集名稱 ('00177' 或 'apidms')
        shot_type (str): shot類型 ('5shots' 或 '10shots')
        model_dir (str): model資料夾路徑
    
    Returns:
        str: Siamese Network權重檔案路徑，如果不存在則返回None
    """
    # 將shot_type轉換為權重檔案格式
    if shot_type == "5shots":
        shot_suffix = "5shot"
    elif shot_type == "10shots":
        shot_suffix = "10shot"
    else:
        print(f"警告: 不支援的shot類型: {shot_type}，將使用隨機初始化的額外層")
        return None
    
    # 構建權重檔案名稱
    weights_filename = f"{dataset_name}_{shot_suffix}_42.pth"
    weights_path = os.path.join(model_dir, weights_filename)
    
    if not os.path.exists(weights_path):
        print(f"警告: 找不到Siamese Network權重檔案: {weights_path}，將使用隨機初始化的額外層")
        return None
    
    return weights_path

def extract_features_for_case(data_root, dataset_name, shot_type, way, case_id, 
                             autoencoder_dir="autoencoder", device='cuda', siamese_weights_path=None, model_dir="model", use_siamese_layers=False):
    """
    為特定case提取特徵
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱
        shot_type (str): shot類型
        way (int): way數量
        case_id (int): case編號
        autoencoder_dir (str): autoencoder資料夾路徑
        device (str): 使用的設備
        siamese_weights_path (str): Siamese Network額外層權重檔案路徑（可選）
        model_dir (str): model資料夾路徑
        use_siamese_layers (bool): 是否使用Siamese Network額外層
    
    Returns:
        dict: 包含support和query特徵的字典
    """
    # 設定圖片轉換 - 調整為64x64以匹配autoencoder期望的輸入尺寸
    # 保持與原始paper一致，不使用正規化
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    # 獲取對應的autoencoder檔案路徑
    autoencoder_path = get_autoencoder_path(dataset_name, autoencoder_dir)
    print(f"使用autoencoder檔案: {autoencoder_path}")
    
    # 如果沒有提供siamese_weights_path，自動獲取對應的權重檔案
    if siamese_weights_path is None:
        siamese_weights_path = get_siamese_weights_path(dataset_name, shot_type, model_dir)
        if siamese_weights_path:
            print(f"自動選擇Siamese Network權重檔案: {siamese_weights_path}")
        else:
            print("將使用隨機初始化的額外層")
    
    # 創建特徵提取器
    extractor = AutoencoderFeatureExtractor(autoencoder_path, device, siamese_weights_path, use_siamese_layers)
    
    # 載入support set
    support_loader = create_lizz_data_loader(
        data_root, dataset_name, shot_type, way, case_id, 
        set_type='known', batch_size=32, transform=transform
    )
    
    # 載入query set
    query_loader = create_lizz_data_loader(
        data_root, dataset_name, shot_type, way, case_id, 
        set_type='unknown', batch_size=32, transform=transform
    )
    
    # 提取support set特徵
    support_features = []
    support_labels = []
    support_class_names = support_loader.dataset.get_class_names()
    
    print(f"正在提取support set特徵 (Case {case_id}, {way}-way)...")
    for images, labels in support_loader:
        images = images.to(device)
        with torch.no_grad():
            features = extractor.extract_features(images)
            support_features.append(features)
            support_labels.append(labels)
    
    support_features = torch.cat(support_features, dim=0)
    support_labels = torch.cat(support_labels, dim=0)
    
    # 檢查support set特徵
    if not torch.isfinite(support_features).all():
        print(f"警告: support set特徵包含非有限值，正在修復...")
        support_features = torch.nan_to_num(support_features, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"Support set特徵形狀: {support_features.shape}, 數值範圍: [{support_features.min():.4f}, {support_features.max():.4f}]")
    
    # 提取query set特徵
    query_features = []
    query_labels = []
    query_class_names = query_loader.dataset.get_class_names()
    
    print(f"正在提取query set特徵 (Case {case_id}, {way}-way)...")
    for images, labels in query_loader:
        images = images.to(device)
        with torch.no_grad():
            features = extractor.extract_features(images)
            query_features.append(features)
            query_labels.append(labels)
    
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    
    # 檢查query set特徵
    if not torch.isfinite(query_features).all():
        print(f"警告: query set特徵包含非有限值，正在修復...")
        query_features = torch.nan_to_num(query_features, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"Query set特徵形狀: {query_features.shape}, 數值範圍: [{query_features.min():.4f}, {query_features.max():.4f}]")
    
    return {
        'support_features': support_features,
        'support_labels': support_labels,
        'support_class_names': support_class_names,
        'query_features': query_features,
        'query_labels': query_labels,
        'query_class_names': query_class_names,
        'case_info': {
            'dataset_name': dataset_name,
            'shot_type': shot_type,
            'way': way,
            'case_id': case_id
        }
    }

def save_case_features(features_dict, output_path):
    """
    保存case特徵到檔案
    
    Args:
        features_dict (dict): 特徵字典
        output_path (str): 輸出檔案路徑
    """
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"特徵已保存到: {output_path}")

def load_case_features(features_path):
    """
    載入case特徵
    
    Args:
        features_path (str): 特徵檔案路徑
    
    Returns:
        dict: 特徵字典
    """
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def batch_extract_features(data_root, dataset_name, shot_type, way, 
                          output_dir, autoencoder_dir="autoencoder", device='cuda'):
    """
    批量提取所有cases的特徵
    
    Args:
        data_root (str): Lizz_data資料夾路徑
        dataset_name (str): 資料集名稱
        shot_type (str): shot類型
        way (int): way數量
        autoencoder_dir (str): autoencoder資料夾路徑
        output_dir (str): 輸出目錄
        device (str): 使用的設備
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"開始批量提取特徵: {dataset_name}, {shot_type}, {way}-way")
    
    for case_id in range(1, 36):  # 35個cases
        print(f"\n處理 Case {case_id}/35...")
        
        try:
            # 提取特徵
            features = extract_features_for_case(
                data_root, dataset_name, shot_type, way, case_id,
                autoencoder_dir, device
            )
            
            # 保存特徵
            output_path = os.path.join(
                output_dir, 
                f"{dataset_name}_{shot_type}_{way}way_case{case_id:02d}.pkl"
            )
            save_case_features(features, output_path)
            
        except Exception as e:
            print(f"Case {case_id} 處理失敗: {e}")
            continue
    
    print(f"\n批量提取完成！特徵保存在: {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 設定路徑
    data_root = "Lizz_data"
    autoencoder_dir = "autoencoder"  # autoencoder資料夾路徑
    output_dir = "lizz_features"
    
    # 提取00177資料集5-shot 5-way的特徵
    batch_extract_features(
        data_root=data_root,
        dataset_name="00177",
        shot_type="5shots",
        way=5,
        autoencoder_dir=autoencoder_dir,
        output_dir=output_dir
    )
