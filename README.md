# CNN 技術指標影像交易策略

使用卷積神經網路（CNN）將技術指標數值轉換為二維影像，並以此進行股市買賣訊號預測與回測。

---

## 專案簡介

本專案以台灣加權指數（^TWII）為標的，透過以下流程建構自動化量化交易策略：

1. 使用 `yfinance` 下載歷史股價資料
2. 利用 `finta` 函式庫計算 15 項技術指標，形成 **15×15 的二維特徵矩陣（影像）**
3. 將此影像輸入預先訓練好的 CNN 模型，預測下一個時間點的操作訊號（持有 / 買入 / 賣出）
4. 根據預測訊號進行模擬回測，並評估策略績效

---

## 技術指標（Features）

每筆樣本由連續 **15 個交易日** × **15 項技術指標** 組成：

| 指標名稱 | 說明 |
|----------|------|
| SMA | 簡單移動平均線 |
| EMA | 指數移動平均線 |
| K | 隨機指標 K 值（Stochastic %K）|
| TEMA | 三重指數移動平均線 |
| D | 隨機指標 D 值（Stochastic %D）|
| RSI | 相對強弱指標 |
| STOCHRSI | 隨機 RSI |
| OBV | 能量潮指標 |
| MACD | MACD 線 − 訊號線 |
| PPO | PPO 線 − 訊號線 |
| ROC | 變動率指標 |
| DMI | DI+ − DI− |
| ADX | 趨向指標 |
| ATR | 平均真實波幅 |
| MOM | 動量指標 |

---

## CNN 模型架構

模型為序列式（Sequential）CNN，輸入形狀為 **(15, 15, 1)**：

```
Layer (type)         Output Shape        Param #
================================================
Conv2D               (None, 13, 13, 32)  320
Conv2D               (None, 11, 11, 64)  18,496
Conv2D               (None,  9,  9, 128) 73,856
Conv2D               (None,  7,  7, 256) 295,168
Flatten              (None, 12544)        0
Dropout              (None, 12544)        0
Dense(3)             (None, 3)           37,635
================================================
Total params: 425,475
```

輸出類別：

| 輸出值 | 意義 |
|--------|------|
| 0 | 觀望（Pass）|
| 1 | 買入（Buy）|
| 2 | 賣出（Sell）|

---

## 回測策略

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `capital` | 1,000,000 | 初始資本（元）|
| `t` | 20 | 最大持有天數（超過則強制出場）|
| `ub` | 1.03 | 止盈倍率（進場價 × 1.03 時賣出）|
| `lb` | 0.97 | 止損倍率（進場價 × 0.97 時賣出）|

回測期間：**2018-01-01 ～ 2023-02-26**（訓練資料未包含的時間段）

---

## 環境需求

- Python 3.10+
- Google Colab（建議，模型與 Scaler 存於 Google Drive）

### 安裝相依套件

```bash
pip install yfinance finta keras tensorflow
```

---

## 使用方式

1. 將預訓練模型 `model_cnn_15_3.h5` 與 `scaler_3.pkl` 上傳至 Google Drive 的 `ChatGPT-AI-InvestmentAdvisor/` 目錄
2. 在 Google Colab 開啟 `CNN_18_TA_startegy.ipynb`
3. 依序執行所有儲存格
4. 執行完畢後，畫面將顯示：
   - 買入訊號圖（紅色向上三角形）
   - 賣出訊號圖（綠色向下三角形）
   - 資本歷史曲線
   - 最終報酬率與最大虧損百分比

---

## 輸出範例

```
最終報酬率: +XX.XX%
最大虧損: -XX.XX%
```

---

## 參考資料

- 技術指標計算：[ChatGPT ShareGPT 對話](https://sharegpt.com/c/CTyoQQ2)
- 回測框架：[ChatGPT ShareGPT 對話](https://shareg.pt/wQwSsaA)
- 原始參考 Notebook：[CNN_15_TA_startegy.ipynb](https://github.com/skywalker0803r/telegram-investment-advice-bot/blob/main/CNN_15_TA_startegy.ipynb)
