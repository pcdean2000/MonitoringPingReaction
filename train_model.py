import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np

# --- 設定 ---
CSV_DATA_FILE = 'ping_data.csv'       # 從此檔案讀取數據
MODEL_FILE_PREFIX = 'iforest_model' # 儲存的模型檔案前綴 (需與主腳本一致)

def train_and_save_models():
    """
    讀取 Ping 數據，為每個目標 IP 訓練一個 Isolation Forest 模型並儲存。
    """
    print("--- Starting AI Model Training ---")
    
    try:
        # 讀取數據並進行基本預處理
        df = pd.read_csv(CSV_DATA_FILE)
        df.dropna(subset=['rtt_ms'], inplace=True) # 移除 ping 失敗的紀錄
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if df.empty:
            print("[!] Error: No valid data with RTT values found in CSV. Cannot train models.")
            return

    except FileNotFoundError:
        print(f"[!] Error: Data file '{CSV_DATA_FILE}' not found. Please run the collection script first.")
        return
    
    # 為每個獨立的 IP 目標進行訓練
    unique_targets = df['target_ip'].unique()
    print(f"Found data for targets: {unique_targets}")

    for target in unique_targets:
        print(f"\n[*] Training model for target: {target}")
        target_df = df[df['target_ip'] == target].copy()
        
        if len(target_df) < 50:
            print(f"  [?] Skipping {target}: Not enough data points ({len(target_df)}). At least 50 are recommended.")
            continue
            
        # 準備特徵 (Features)
        # 我們使用 RTT 和「一天中的小時」作為特徵。
        # 加入「小時」可以讓模型學會一天中不同時段的正常 RTT 變化 (例如白天和半夜的網路負載可能不同)
        target_df['hour_of_day'] = target_df['timestamp'].dt.hour
        features = target_df[['rtt_ms', 'hour_of_day']].values
        
        # 初始化並訓練 Isolation Forest 模型
        # 'contamination' 設為 'auto'，讓演算法自行決定數據中的異常比例
        # 'random_state' 確保每次訓練結果可重現
        model = IsolationForest(contamination='auto', random_state=42)
        model.fit(features)
        
        # 儲存訓練好的模型
        model_filename = f"{MODEL_FILE_PREFIX}_{target.replace('.', '_')}.joblib"
        joblib.dump(model, model_filename)
        print(f"  [+] Model trained and saved as '{model_filename}'")
        
    print("\n--- Model Training Finished ---")

if __name__ == "__main__":
    train_and_save_models()