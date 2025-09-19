# Lizz_data Few-Shot Learning 評估工具

這個工具包專門用於評估您的autoencoder.pth檔案在Lizz_data資料集上的few-shot learning表現。

## 資料集結構

Lizz_data包含兩個資料集：
- **00177**: 第一個資料集 (對應autoencoder: `split3_42_autoencoder_model.pth`)
- **apidms**: 第二個資料集 (對應autoencoder: `apidms_42_autoencoder_model.pth`)

### Autoencoder對應關係
- **00177** 資料集使用 `autoencoder/split3_42_autoencoder_model.pth`
- **apidms** 資料集使用 `autoencoder/apidms_42_autoencoder_model.pth`

工具會自動根據資料集名稱選擇對應的autoencoder檔案。

每個資料集包含：
- **5shots_5_10_35cases**: 5-shot設定，支援5-way到10-way，每個35個cases
- **10shots_5_10_35cases**: 10-shot設定，支援5-way到10-way，每個35個cases

### 資料夾結構
```
Lizz_data/
├── 00177_5shots_5_10_35cases/
│   ├── known/                    # 基礎已知類別圖片
│   ├── known_with_case/          # 按case組織的support set
│   │   ├── test_5_1/            # 5-way第1個case
│   │   ├── test_5_2/            # 5-way第2個case
│   │   ├── ...
│   │   ├── test_10_1/           # 10-way第1個case
│   │   └── test_10_35/          # 10-way第35個case
│   ├── unknown/                  # 基礎未知類別圖片
│   └── unknown_with_case/        # 按case組織的query set
│       ├── test_5_1/            # 5-way第1個case的query set
│       ├── test_5_2/            # 5-way第2個case的query set
│       ├── ...
│       ├── test_10_1/           # 10-way第1個case的query set
│       └── test_10_35/          # 10-way第35個case的query set
└── apidms_5shots_5_10_35cases/
    └── ... (相同結構)
```

### Case命名規則
- **test_5_1** 到 **test_5_35**: 5-way的35個cases
- **test_6_1** 到 **test_6_35**: 6-way的35個cases
- **test_7_1** 到 **test_7_35**: 7-way的35個cases
- **test_8_1** 到 **test_8_35**: 8-way的35個cases
- **test_9_1** 到 **test_9_35**: 9-way的35個cases
- **test_10_1** 到 **test_10_35**: 10-way的35個cases

## 檔案說明

1. **lizz_data_loader.py**: Lizz_data資料載入器
2. **test_lizz_data_evaluation.py**: 主要的評估腳本
3. **run_lizz_evaluation.py**: 簡化的使用界面

## 使用方法

### 方法1: 使用簡化界面（推薦）

```bash
python run_lizz_evaluation.py
```

工具會自動：
- 檢查Lizz_data資料夾是否存在
- 檢查autoencoder資料夾中的對應檔案
- 根據資料集名稱自動選擇正確的autoencoder檔案

### 方法2: 直接使用評估腳本

```python
from test_lizz_data_evaluation import evaluate_lizz_35_cases

# 評估特定設定
results = evaluate_lizz_35_cases(
    data_root="Lizz_data",
    dataset_name="00177",
    shot_type="5shots",
    way=5,
    autoencoder_dir="autoencoder"  # 自動選擇對應的autoencoder檔案
)
```

### 方法3: 批量評估所有設定

```python
from test_lizz_data_evaluation import batch_evaluate_lizz_data

# 評估所有8個設定
batch_evaluate_lizz_data(
    data_root="Lizz_data",
    autoencoder_dir="autoencoder"  # 自動選擇對應的autoencoder檔案
)
```

## 支援的評估設定

工具支援以下8個設定的評估：

1. **00177 5shots 5-way** - 35個cases
2. **00177 5shots 10-way** - 35個cases
3. **00177 10shots 5-way** - 35個cases
4. **00177 10shots 10-way** - 35個cases
5. **apidms 5shots 5-way** - 35個cases
6. **apidms 5shots 10-way** - 35個cases
7. **apidms 10shots 5-way** - 35個cases
8. **apidms 10shots 10-way** - 35個cases

## 輸出結果

評估完成後，您將得到：

1. **控制台輸出**: 顯示35個cases的平均指標
2. **詳細結果檔案**: `lizz_evaluation_{dataset}_{shot_type}_{way}way.pkl`
3. **所有結果檔案**: `all_lizz_evaluation_results.pkl`（批量評估時）

### 指標說明

- **準確率 (Accuracy)**: 正確預測的比例
- **精確率 (Precision)**: 預測為正類的樣本中實際為正類的比例
- **召回率 (Recall)**: 實際正類樣本中被正確預測的比例
- **F1分數 (F1-Score)**: 精確率和召回率的調和平均
- **PR-AUC**: Precision-Recall曲線下面積

## 範例輸出

```
評估Lizz_data: 00177, 5shots, 5-way
============================================================
正在提取support set特徵 (Case 1, 5-way)...
正在提取query set特徵 (Case 1, 5-way)...
...

============================================================
35個Cases的平均結果 (00177, 5shots, 5-way):
============================================================
準確率 (Accuracy): 0.7234 ± 0.0456
精確率 (Precision): 0.7156 ± 0.0523
召回率 (Recall): 0.7234 ± 0.0456
F1分數 (F1-Score): 0.7189 ± 0.0489
PR-AUC: 0.7891 ± 0.0345
============================================================
```

## 自定義autoencoder架構

如果您的autoencoder架構與預設的不同，請修改 `autoencoder_feature_extractor.py` 中的 `_create_model_architecture` 方法。

## 注意事項

1. **圖片格式**: 支援PNG格式的圖片
2. **記憶體使用**: 大型資料集可能需要較多記憶體
3. **GPU支援**: 自動檢測並使用可用的GPU
4. **資料路徑**: 確保Lizz_data資料夾在正確的位置
5. **Autoencoder對應**: 工具會自動根據資料集名稱選擇對應的autoencoder檔案

## 故障排除

### 常見問題

1. **資料夾不存在**
   - 檢查Lizz_data資料夾路徑是否正確
   - 確認資料夾結構是否完整

2. **圖片載入失敗**
   - 檢查圖片格式是否為PNG
   - 確認圖片檔案沒有損壞

3. **模型載入失敗**
   - 檢查autoencoder資料夾是否存在
   - 確認對應的autoencoder檔案是否存在
   - 確認模型格式是否正確

4. **記憶體不足**
   - 減少batch_size
   - 使用CPU而非GPU

### 調試模式

在評估函數中設定 `verbose=True` 可以查看詳細的評估過程：

```python
metrics = evaluate_single_lizz_case(
    data_root, dataset_name, shot_type, way, case_id,
    autoencoder_path, verbose=True  # 啟用詳細輸出
)
```

## 與原始FSLTask.py的整合

這個工具完全兼容您現有的FSLTask.py，使用相同的：
- 數據預處理步驟
- 模型架構
- 優化算法
- 評估指標

唯一的差別是資料載入方式，專門針對Lizz_data的結構進行了優化。

## 支援

如果您遇到任何問題，請檢查：
1. Lizz_data資料夾結構是否正確
2. autoencoder資料夾是否存在且包含對應的.pth檔案
3. 依賴套件是否已安裝
4. GPU記憶體是否足夠

祝您使用愉快！
