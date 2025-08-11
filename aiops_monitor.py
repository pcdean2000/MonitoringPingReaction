
import subprocess
import time
import datetime
import re
import platform
import os
import pandas as pd
import joblib
import numpy as np
import requests
import configparser

# ==============================================================================
# --- 1. 全域設定 (可依需求修改) ---
# ==============================================================================
# 讀取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 監控的目標 IP (建議包含穩定目標、備用目標及一個無效 IP 以測試丟包)
TARGETS = ['8.8.8.8', '1.1.1.1'] 
PING_INTERVAL_SECONDS = 10  # 每 10 秒執行一次 PING
CSV_DATA_FILE = 'ping_data.csv'  # 儲存歷史數據的檔案
MODEL_FILE_PREFIX = 'iforest_model' # 模型檔案名稱的前綴

# --- Telegram Bot 設定 (從 config.ini 讀取) ---
TELEGRAM_ENABLED = config.getboolean('TELEGRAM', 'TELEGRAM_ENABLED', fallback=False)

if TELEGRAM_ENABLED:
    TELEGRAM_BOT_TOKEN = config.get('TELEGRAM', 'TELEGRAM_BOT_TOKEN', fallback="YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = config.get('TELEGRAM', 'TELEGRAM_CHAT_ID', fallback="YOUR_TELEGRAM_CHAT_ID")
else:
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# --- AI 與決策規則設定 ---
# 連續幾次丟包率超過閾值才觸發「連線中斷」警報
PACKET_LOSS_THRESHOLD_PERCENT = 50.0 
CONSECUTIVE_LOSS_TRIGGER = 3

# ==============================================================================
# --- 2. 數據收集與解析 (Data Collection) ---
# ==============================================================================
def parse_ping_output(output: str) -> (float, float):
    """
    根據不同作業系統解析 ping 指令的輸出，提取 RTT (平均來回時間) 和封包遺失率。
    :param output: ping 指令的 stdout/stderr 內容。
    :return: (rtt_ms, packet_loss_percent) Tuple。若無法解析則為 (None, 100.0)。
    """
    os_type = platform.system().lower()
    rtt_ms, packet_loss = None, 100.0

    try:
        if "windows" in os_type:
            rtt_match = re.search(r"Average = (\d+)ms", output)
            if rtt_match:
                rtt_ms = float(rtt_match.group(1))
            loss_match = re.search(r"\((\d+)% loss\)", output)
            if loss_match:
                packet_loss = float(loss_match.group(1))
        else:  # Linux and macOS
            rtt_match = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", output)
            if rtt_match:
                rtt_ms = float(rtt_match.group(1))
            loss_match = re.search(r"(\d+)% packet loss", output)
            if loss_match:
                packet_loss = float(loss_match.group(1))
    except (IndexError, TypeError):
        # 如果解析出錯，回傳最壞情況
        return None, 100.0
    
    # 如果 ping 一個不存在的主機，可能沒有 RTT 但有 100% 丟包率
    if rtt_ms is None and packet_loss == 100.0:
        return None, 100.0
        
    return rtt_ms, packet_loss

def execute_ping(target: str) -> (float, float):
    """
    執行單次 ping 命令。
    :param target: 要 ping 的 IP 位址。
    :return: (rtt_ms, packet_loss_percent) Tuple。
    """
    os_type = platform.system().lower()
    # 根據作業系統設定 ping 指令參數 (發送 4 個封包)
    command = ['ping', '-n' if 'windows' in os_type else '-c', '4', target]
    
    try:
        # 執行指令，設定 15 秒超時以防卡住
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if 'windows' in os_type else 0 # Windows下不顯示黑窗
        )
        return parse_ping_output(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        print(f"  [!] Ping to {target} timed out.")
        return None, 100.0
    except Exception as e:
        print(f"  [!] An error occurred while pinging {target}: {e}")
        return None, 100.0

# ==============================================================================
# --- 3. AI 分析與決策 (AI Analysis & Decision) ---
# ==============================================================================
# 全域變數，用於儲存載入的模型和追蹤連續異常狀態
loaded_models = {}
consecutive_loss_counters = {target: 0 for target in TARGETS}

def load_ai_models():
    """啟動時載入所有已訓練好的模型到記憶體中。"""
    print("\n--- Loading AI Models ---")
    for target in TARGETS:
        model_filename = f"{MODEL_FILE_PREFIX}_{target.replace('.', '_')}.joblib"
        if os.path.exists(model_filename):
            try:
                loaded_models[target] = joblib.load(model_filename)
                print(f"  [+] Model for {target} loaded successfully.")
            except Exception as e:
                print(f"  [!] Failed to load model for {target}: {e}")
        else:
            print(f"  [?] Warning: Model for {target} ('{model_filename}') not found. Will skip AI detection for this target.")
    print("-------------------------\n")

def check_rtt_anomaly(target: str, rtt: float, timestamp: datetime.datetime) -> (bool, str):
    """
    使用載入的 AI 模型檢查 RTT 是否為異常。
    :param target: 目標 IP。
    :param rtt: 當前的 RTT 值。
    :param timestamp: 當前時間戳，用於提取特徵 (如小時)。
    :return: (is_anomaly, reason) Tuple。
    """
    if target not in loaded_models:
        return False, "No model available."

    # 準備特徵，需與訓練時的特徵完全一致
    # 這裡我們使用 RTT 和「一天中的小時」作為特徵
    features = np.array([[rtt, timestamp.hour]])
    
    # 進行預測，-1 代表異常 (outlier)，1 代表正常 (inlier)
    prediction = loaded_models[target].predict(features)
    
    if prediction[0] == -1:
        return True, f"AI model detected anomalous RTT ({rtt:.2f}ms)."
    
    return False, "RTT is within normal range."

# ==============================================================================
# --- 4. 反應執行 (Reaction Execution) ---
# ==============================================================================
def send_telegram_alert(message: str):
    """
    透過 Telegram Bot API 發送格式化的警告訊息。
    """
    if not TELEGRAM_ENABLED:
        print("  [i] Telegram notifications are disabled in config.")
        return
        
    if not all([TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN", TELEGRAM_CHAT_ID != "YOUR_TELEGRAM_CHAT_ID"]):
        print("  [!] Telegram credentials are not configured. Skipping notification.")
        return

    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'  # 使用 Markdown 格式化訊息
    }
    
    try:
        response = requests.post(api_url, data=payload, timeout=10)
        if response.status_code == 200:
            print("  [+] Alert sent to Telegram successfully.")
        else:
            print(f"  [!] Failed to send Telegram alert. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"  [!] Exception while sending Telegram alert: {e}")

# ==============================================================================
# --- 5. 主執行迴圈 (Main Loop) ---
# ==============================================================================
def main():
    """主執行函式。"""
    # 檢查 CSV 檔案是否存在，若否，則創建並寫入檔頭
    if not os.path.exists(CSV_DATA_FILE):
        with open(CSV_DATA_FILE, 'w', encoding='utf-8') as f:
            f.write('timestamp,target_ip,rtt_ms,packet_loss_percent\n')

    # 啟動時載入 AI 模型
    load_ai_models()
    
    # 發送啟動通知
    startup_message = "🚀 *AIOps 網路監控系統已啟動* 🚀\n開始監控目標: `" + ", ".join(TARGETS) + "`"
    print(startup_message)
    send_telegram_alert(startup_message)
    
    try:
        while True:
            for target in TARGETS:
                current_time = datetime.datetime.now()
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp_str}] Pinging {target}...")

                # 1. 數據收集
                rtt, packet_loss = execute_ping(target)
                
                if rtt is not None:
                    print(f"  -> Result: RTT={rtt:.2f}ms, Packet Loss={packet_loss}%")
                else:
                    print(f"  -> Result: Failed to get RTT, Packet Loss={packet_loss}%")

                # 將結果寫入 CSV
                with open(CSV_DATA_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{current_time.isoformat()},{target},{rtt if rtt is not None else ''},{packet_loss}\n")

                # 2. 決策與反應
                # 規則一：檢查連線中斷
                if packet_loss >= PACKET_LOSS_THRESHOLD_PERCENT:
                    consecutive_loss_counters[target] += 1
                else:
                    consecutive_loss_counters[target] = 0 # 只要成功一次就重置計數器

                if consecutive_loss_counters[target] >= CONSECUTIVE_LOSS_TRIGGER:
                    alert_msg = (
                        f"🚨 *[網路中斷警報]* 🚨\n\n"
                        f"*目標:* `{target}`\n"
                        f"*狀態:* 連續 {CONSECUTIVE_LOSS_TRIGGER} 次封包遺失率 ≥ {PACKET_LOSS_THRESHOLD_PERCENT}%\n"
                        f"*目前遺失率:* {packet_loss}%\n"
                        f"*時間:* {timestamp_str}"
                    )
                    send_telegram_alert(alert_msg)
                    consecutive_loss_counters[target] = 0 # 發送後重置，避免重複洗版
                
                # 規則二：若連線正常，則檢查 RTT 異常
                elif rtt is not None:
                    is_anomaly, reason = check_rtt_anomaly(target, rtt, current_time)
                    if is_anomaly:
                        alert_msg = (
                            f"🟠 *[網路延遲警報]* 🟠\n\n"
                            f"*目標:* `{target}`\n"
                            f"*狀態:* {reason}\n"
                            f"*時間:* {timestamp_str}"
                        )
                        send_telegram_alert(alert_msg)
            
            # 等待指定間隔
            time.sleep(PING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        shutdown_message = "👋 *AIOps 網路監控系統已手動停止* 👋"
        print(f"\n{shutdown_message}")
        send_telegram_alert(shutdown_message)
    except Exception as e:
        error_message = f"🔥 *AIOps 監控系統發生嚴重錯誤* 🔥\n\n*錯誤詳情:* `{str(e)}`\n程式已終止，請立即檢查！"
        print(f"\n{error_message}")
        send_telegram_alert(error_message)

if __name__ == "__main__":
    main()