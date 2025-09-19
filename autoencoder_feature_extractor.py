import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os

class AutoencoderFeatureExtractor:
    """
    Autoencoder特徵提取器，用於從.pth檔案載入訓練好的autoencoder模型並提取特徵
    """
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', siamese_weights_path=None, use_siamese_layers=False):
        """
        初始化特徵提取器
        
        Args:
            model_path (str): autoencoder.pth檔案路徑
            device (str): 使用的設備 ('cuda' 或 'cpu')
            siamese_weights_path (str): Siamese Network額外層權重檔案路徑（可選）
        """
        self.device = device
        self.model = None
        self.siamese_weights_path = siamese_weights_path
        self.added_layers = None
        self.use_siamese_layers = use_siamese_layers
        self.load_model(model_path)
        if use_siamese_layers:
            self.load_siamese_weights()
        
    def load_model(self, model_path):
        """
        載入autoencoder模型
        
        Args:
            model_path (str): .pth檔案路徑
        """
        try:
            # 載入模型檢查點
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 根據不同的模型結構進行載入
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 如果檢查點包含model_state_dict
                    self.model = self._create_model_architecture(checkpoint)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    # 如果檢查點包含state_dict
                    self.model = self._create_model_architecture(checkpoint)
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 如果檢查點直接是state_dict (OrderedDict)
                    self.model = self._create_model_architecture(checkpoint)
                    self.model.load_state_dict(checkpoint)
            else:
                # 如果檢查點直接是模型
                self.model = checkpoint
                
            # 確保模型在正確的設備上
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            print(f"成功載入autoencoder模型: {model_path}")
            
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            print("請確認模型檔案路徑和格式是否正確")
            raise e
    
    def load_siamese_weights(self):
        """
        載入Siamese Network額外層權重
        """
        if self.siamese_weights_path is None:
            print("未提供Siamese Network權重路徑，將使用隨機初始化的額外層")
            return
        
        try:
            if not os.path.exists(self.siamese_weights_path):
                print(f"Siamese Network權重檔案不存在: {self.siamese_weights_path}")
                print("將使用隨機初始化的額外層")
                return
            
            # 載入權重
            siamese_weights = torch.load(self.siamese_weights_path, map_location=self.device)
            print(f"成功載入Siamese Network權重: {self.siamese_weights_path}")
            
            # 創建額外層架構
            self.added_layers = nn.Sequential(
                nn.Linear(64, 250), 
                nn.ReLU(inplace=True),
                nn.Linear(250, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, 150)
            ).to(self.device)
            
            # 載入權重到額外層
            if isinstance(siamese_weights, dict):
                # 如果是state_dict格式
                if 'added_layers' in siamese_weights:
                    self.added_layers.load_state_dict(siamese_weights['added_layers'])
                elif 'model_state_dict' in siamese_weights:
                    # 嘗試從完整模型中提取額外層權重
                    model_state = siamese_weights['model_state_dict']
                    added_layers_state = {k.replace('added_layers.', ''): v 
                                        for k, v in model_state.items() 
                                        if k.startswith('added_layers.')}
                    if added_layers_state:
                        self.added_layers.load_state_dict(added_layers_state)
                    else:
                        print("無法從權重檔案中找到額外層權重，使用隨機初始化")
                else:
                    # 嘗試直接載入 - 檢查是否有added_layers前綴
                    added_layers_state = {k.replace('added_layers.', ''): v 
                                        for k, v in siamese_weights.items() 
                                        if k.startswith('added_layers.')}
                    if added_layers_state:
                        self.added_layers.load_state_dict(added_layers_state)
                    else:
                        print("權重檔案格式不匹配，使用隨機初始化")
            else:
                # 如果是完整的模型
                if hasattr(siamese_weights, 'added_layers'):
                    self.added_layers.load_state_dict(siamese_weights.added_layers.state_dict())
                else:
                    print("權重檔案格式不匹配，使用隨機初始化")
            
            # 凍結額外層參數
            for param in self.added_layers.parameters():
                param.requires_grad = False
                
            print("Siamese Network額外層權重載入完成")
            
        except Exception as e:
            print(f"載入Siamese Network權重時發生錯誤: {e}")
            print("將使用隨機初始化的額外層")
            self.added_layers = None
    
    def _create_model_architecture(self, checkpoint):
        """
        根據檢查點創建模型架構
        這裡需要根據您的autoencoder架構進行調整
        """
        # 使用正確的Autoencoder_Conv架構 - 針對64x64輸入調整
        class Autoencoder_Conv(nn.Module):
            def __init__(self):
                super().__init__()
                # N, 3, 64, 64 (輸入是64x64的RGB圖像)
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> N, 16, 32, 32
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 16, 16
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 16)  # -> N, 64, 1, 1
                )

                # N , 64, 1, 1
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 16),  # -> N, 32, 16, 16
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                    # N, 16, 32, 32
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # N, 3, 64, 64
                    nn.Sigmoid()
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded  # 返回 (decoded, encoded) 以便特徵提取
            
            def encode(self, x):
                return self.encoder(x)
        
        # 檢查是否為CNN架構
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            
            # 檢查是否有Conv2d層
            conv_keys = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 4]
            if conv_keys:
                print("檢測到CNN-based autoencoder架構")
                return Autoencoder_Conv()
        
        # 如果無法推斷，使用預設的CNN架構
        print("使用預設CNN autoencoder架構")
        return Autoencoder_Conv()
    
    def extract_features(self, data):
        """
        從數據中提取特徵，模擬Siamese Network的處理流程
        
        Args:
            data (torch.Tensor): 輸入數據，形狀為 (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: 提取的特徵，形狀為 (batch_size, encoded_dim)
        """
        if self.model is None:
            raise ValueError("模型尚未載入，請先載入模型")
        
        with torch.no_grad():
            data = data.to(self.device)
            
            # 檢查輸入數據是否包含非有限值
            if not torch.isfinite(data).all():
                print(f"警告: 輸入數據包含非有限值 (NaN/Inf)")
                data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 如果模型有encode方法，直接使用
            if hasattr(self.model, 'encode'):
                features = self.model.encode(data)
            else:
                # 否則嘗試從forward方法中獲取編碼特徵
                try:
                    _, features = self.model(data)
                except:
                    # 如果模型沒有返回編碼特徵，可能需要其他處理方式
                    features = self.model(data)
            
            # 檢查特徵是否包含非有限值
            if not torch.isfinite(features).all():
                print(f"警告: 提取的特徵包含非有限值 (NaN/Inf)")
                features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 將CNN特徵圖展平為向量
            if len(features.shape) > 2:  # 如果是特徵圖 (batch_size, channels, height, width)
                features = features.view(features.size(0), -1)  # 展平為 (batch_size, channels*height*width)
            
            # 使用Siamese Network的額外層處理（可選）
            if self.use_siamese_layers and features.size(1) == 64:  # 確保是64維特徵
                if self.added_layers is not None:
                    # 使用預訓練的額外層
                    features = self.added_layers(features)
                    print(f"使用預訓練Siamese Network額外層處理特徵: {features.shape}")
                else:
                    # 創建隨機初始化的額外層
                    added_layers = torch.nn.Sequential(
                        torch.nn.Linear(64, 250), 
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(250, 200),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(200, 150)
                    ).to(self.device)
                    
                    # 使用額外層處理特徵
                    features = added_layers(features)
                    print(f"使用隨機初始化額外層處理特徵: {features.shape}")
                
                # 不進行正規化，讓PCA降維自己處理
                # features = torch.nn.functional.normalize(features, p=2, dim=1)
            else:
                print(f"跳過Siamese Network額外層處理，直接使用原始特徵: {features.shape}")
            
            # 最終檢查並清理特徵
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features.cpu()
    
    def extract_features_from_dataset(self, dataset_path):
        """
        從數據集檔案中提取特徵
        
        Args:
            dataset_path (str): 數據集檔案路徑（.plk或.pickle檔案）
            
        Returns:
            dict: 包含特徵和標籤的字典
        """
        import pickle
        
        # 載入原始數據集
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            if data.__len__() == 2:
                data = data[1]
        
        # 提取特徵
        features_dict = {}
        for class_key, class_data in data.items():
            print(f"正在處理類別 {class_key}，包含 {len(class_data)} 個樣本")
            
            # 將類別數據轉換為tensor
            class_tensor = torch.FloatTensor(np.stack(class_data, axis=0))
            
            # 分批處理以避免記憶體問題
            batch_size = 1000
            features_list = []
            
            for i in range(0, len(class_tensor), batch_size):
                batch = class_tensor[i:i+batch_size]
                batch_features = self.extract_features(batch)
                features_list.append(batch_features)
            
            # 合併所有批次的特徵
            features_dict[class_key] = torch.cat(features_list, dim=0)
        
        return features_dict

def create_autoencoder_dataset_file(features_dict, output_path):
    """
    將提取的特徵保存為與FSLTask.py兼容的格式
    
    Args:
        features_dict (dict): 包含特徵的字典
        output_path (str): 輸出檔案路徑
    """
    import pickle
    
    # 創建與FSLTask.py兼容的數據格式
    dataset = {
        'data': features_dict,
        'labels': None  # 標籤將在FSLTask.py中生成
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"特徵已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 設定路徑
    autoencoder_path = "path/to/your/autoencoder.pth"  # 替換為您的autoencoder.pth路徑
    dataset_path = "path/to/your/dataset.plk"  # 替換為您的數據集路徑
    output_path = "autoencoder_features.plk"  # 輸出特徵檔案路徑
    
    # 創建特徵提取器
    extractor = AutoencoderFeatureExtractor(autoencoder_path)
    
    # 提取特徵
    features_dict = extractor.extract_features_from_dataset(dataset_path)
    
    # 保存特徵
    create_autoencoder_dataset_file(features_dict, output_path)
