# Siamese Network權重載入說明

## 概述

現在系統支持載入預訓練的Siamese Network額外層權重，以確保特徵提取與其他方法保持一致。

## 使用方法

### 1. 準備權重檔案

請將你的Siamese Network權重檔案放在適當的位置，例如：
```
autoencoder/
├── split3_42_autoencoder_model.pth
├── apidms_42_autoencoder_model.pth
└── siamese_weights.pth  # 你的Siamese Network權重檔案
```

### 2. 權重檔案格式

系統支持以下格式的權重檔案：

#### 格式1：只包含額外層權重
```python
{
    'added_layers': {
        '0.weight': tensor(...),
        '0.bias': tensor(...),
        '2.weight': tensor(...),
        '2.bias': tensor(...),
        ...
    }
}
```

#### 格式2：完整模型state_dict
```python
{
    'model_state_dict': {
        'added_layers.0.weight': tensor(...),
        'added_layers.0.bias': tensor(...),
        'added_layers.2.weight': tensor(...),
        'added_layers.2.bias': tensor(...),
        ...
    }
}
```

#### 格式3：直接state_dict
```python
{
    '0.weight': tensor(...),
    '0.bias': tensor(...),
    '2.weight': tensor(...),
    '2.bias': tensor(...),
    ...
}
```

### 3. 修改評估腳本

在 `test_lizz_data_evaluation_unbalanced.py` 中修改：

```python
# 設定Siamese Network權重路徑
siamese_weights_path = "autoencoder/siamese_weights.pth"  # 你的權重檔案路徑

# 在evaluate_single_lizz_case_unbalanced函數中
features_dict = extract_features_for_case(
    data_root, dataset_name, shot_type, way, case_id, 
    autoencoder_dir, siamese_weights_path=siamese_weights_path
)
```

### 4. 測試權重載入

運行測試腳本：
```bash
python test_siamese_weights.py
```

## 權重檔案結構

Siamese Network額外層的架構為：
```python
nn.Sequential(
    nn.Linear(64, 250), 
    nn.ReLU(inplace=True),
    nn.Linear(250, 200),
    nn.ReLU(inplace=True),
    nn.Linear(200, 150)
)
```

## 注意事項

1. **權重檔案不存在**：如果權重檔案不存在或載入失敗，系統會自動使用隨機初始化的額外層
2. **權重格式不匹配**：如果權重格式不匹配，系統會嘗試多種格式，失敗時使用隨機初始化
3. **設備兼容性**：權重會自動載入到正確的設備（CPU/GPU）

## 優勢

使用預訓練的Siamese Network權重可以：
- 確保特徵提取與其他方法完全一致
- 避免隨機初始化帶來的不確定性
- 提高評估結果的可重現性
- 更好地模擬原始論文的實驗設置
