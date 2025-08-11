# MonitoringPingReaction

AIOps 網路監控系統，結合 AI 模型自動偵測網路異常，並可選擇透過 Telegram Bot 發送即時警報。

## 特色
- 定時對多個目標 IP 進行 ping 測試，收集 RTT 與封包遺失率
- 使用 Isolation Forest AI 模型自動判斷網路延遲異常
- 支援 Telegram Bot 警報通知（可開關）
- 所有監控數據自動儲存於 CSV 檔案

## 安裝需求
- Python 3.12 以上
- 依賴套件：
  - pandas
  - numpy
  - requests
  - joblib
  - scikit-learn

安裝依賴：
```bash
pip install -r requirements.txt
```
或
```bash
pip install pandas numpy requests joblib scikit-learn
```

## 使用說明

### 1. 設定 config.ini
請參考 `config.ini.sample` 建立並編輯 `config.ini`：
```ini
[TELEGRAM]
TELEGRAM_ENABLED = true           # 啟用 Telegram 警報 (true/false)
TELEGRAM_BOT_TOKEN = <你的 Bot Token>
TELEGRAM_CHAT_ID = <你的 Chat ID>
```

### 2. 執行監控主程式
```bash
python aiops_monitor.py
```

### 3. 訓練 AI 模型（可選）
收集一段時間的 ping 數據後，可執行：
```bash
python train_model.py
```
會自動為每個目標 IP 訓練 Isolation Forest 模型。

## 檔案說明
- `aiops_monitor.py`：主監控程式，負責數據收集、異常判斷與警報
- `train_model.py`：AI 模型訓練腳本
- `config.ini`：主要設定檔
- `ping_data.csv`：歷史監控數據
- `iforest_model_*.joblib`：訓練好的 AI 模型檔案

## 參數調整
- 監控目標、間隔、警報門檻等可於 `aiops_monitor.py` 內部調整

## License
MIT License
