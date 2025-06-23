# app.py
# Root API: https://binkhoale1812-poptech-cleaner.hf.space/
# Usages: 
## https://binkhoale1812-poptech-cleaner.hf.space/fetch?Password=...
## https://binkhoale1812-poptech-cleaner.hf.space/load?Password=...
## https://binkhoale1812-poptech-cleaner.hf.space/delete?Password=...

import os, json, signal, logging, threading, time
from datetime import datetime, timedelta
from collections import deque 

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from pymongo import MongoClient, UpdateOne
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# ─────── ENV CONFIG ───────
load_dotenv()

BROKER       = os.getenv("BROKER")
PORT         = int(os.getenv("PORT", 1883))
USERNAME     = os.getenv("USERNAME")
PASSWORD     = os.getenv("PASSWORD")
MQTT_TOPIC   = os.getenv("MQTT_TOPIC", "device/socket/reply/#")

MONGO_URI    = os.getenv("MONGO_URI")
MONGO_DB     = os.getenv("MONGO_DB", "poptech")
MONGO_COL    = os.getenv("MONGO_COLLECTION", "device_clean")
FETCH_PASS   = os.getenv("FETCH_PASSWORD")

# Tham số xử lý (thời gian)
EXPECTED_INTERVAL_SEC = int(os.getenv("EXPECTED_INTERVAL_SEC", 30))
TOLERANCE_SEC         = int(os.getenv("TOLERANCE_SEC", 10))
BUFFER_SECONDS        = int(os.getenv("BUFFER_SECONDS", 4 * 3600))   # 4 giờ
BACKFILL_INTERVAL     = int(os.getenv("BACKFILL_INTERVAL", 10))      # 10 giây

RAW_CHECKPOINT_PATH   = os.getenv("RAW_CHECKPOINT_PATH", "cache/checkpoint_raw.csv")
EXPORT_CSV_PATH       = "mongo_cleaned_export.csv"
os.makedirs(os.path.dirname(RAW_CHECKPOINT_PATH), exist_ok=True)

# ─────────────── LOGGING ───────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    force=True
)
logger = logging.getLogger("poptech-cleaner")

# ─────────────── GLOBALS ───────────────
win_len      = BUFFER_SECONDS // EXPECTED_INTERVAL_SEC + 200
window       = deque(maxlen=win_len)           # lưu 4 giờ gần nhất
stop_event   = threading.Event()
app          = FastAPI()

# ─────────────── UTILITIES ───────────────
# Đảm bảo giá trị là float, nếu không flag NaN
def safe_float(x):
    try: return float(x)
    except: return np.nan

def parse_row(ts: str, topic: str, payload: str):
    """Trả về dict đã parse hoặc None nếu không hợp lệ."""
    try:
        j = json.loads(payload.replace('""', '"'))
        if not topic.startswith("device/socket/reply/"):
            return None
        if not isinstance(j.get("data", []), list):
            return None
        v, a, w, c = (j["data"] + [None] * 4)[:4]
        # bỏ frame idle (all 0)
        if all(x in (0, None) for x in (a, w, c)):
            return None
        return {
            "timestamp": ts,
            "id": j.get("id"),
            "imei": j.get("imei"),
            "type": j.get("type"),
            "voltage": safe_float(v),
            "current": safe_float(a),
            "power": safe_float(w),
            "consume": safe_float(c)
        }
    except Exception:
        return None

# Tải dữ liệu mới lên DB
def upsert_mongo(docs):
    if not docs:
        return
    try:
        client = MongoClient(MONGO_URI)
        col    = client[MONGO_DB][MONGO_COL]
        col.create_index("timestamp", unique=True)
        ops = [UpdateOne({"_id": d["timestamp"]}, {"$set": d}, upsert=True) for d in docs]
        col.bulk_write(ops, ordered=False)
    except Exception as e:
        logger.error(f"❌ Mongo error: {e}")

# Chèn giá trị tổng thể
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    # Tổng thời gian dự kiến giữa session
    expected = timedelta(seconds=EXPECTED_INTERVAL_SEC)
    tol      = timedelta(seconds=TOLERANCE_SEC)
    # Lọc lỗi và trống
    rows = [df.iloc[0]]
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1]["timestamp"], df.iloc[i]["timestamp"]
        rows.append(df.iloc[i])
        if curr - prev > expected + tol:
            for j in range(1, int(round((curr - prev) / expected))):
                gap_ts = prev + j * expected
                gap = df.iloc[i-1].copy()
                gap["timestamp"] = gap_ts
                for col in ["voltage", "current", "power", "consume"]:
                    gap[col] = np.nan
                rows.insert(-1, gap)
    # Sort với ts là identifier
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["consume_clean"] = df["consume"]
    df.loc[(df["consume"] < 0) | (df["consume"].diff() < 0), "consume_clean"] = np.nan
    # Impute 3 giá trị còn lại với KNNImputer
    non_missing = df[["voltage","current","power"]].dropna().shape[0]
    k = min(3, max(1, non_missing))
    imputer = KNNImputer(n_neighbors=k)
    df[["voltage", "current", "power"]] = imputer.fit_transform(df[["voltage", "current", "power"]])
    # Train và pred fit với LinearRegression
    train = df[df["consume_clean"].notna()]
    pred  = df[df["consume_clean"].isna()]
    if not train.empty and not pred.empty:
        model = LinearRegression().fit(train[["voltage","current","power"]], train["consume_clean"])
        try:
            y_hat = model.predict(pred[["voltage","current","power"]])
            df.loc[pred.index, "consume_clean"] = pd.Series(y_hat, index=pred.index)
        except Exception as e:
            logger.warning(f"⚠️ Primary model error: {e}")
    # Nếu còn giá trị trống sau bộ lọc đầu, tái sd LinearRegression và dự đoán trên ts + tổng tg giữa session
    still = df[df["consume_clean"].isna()]
    if not still.empty:
        logger.warning(f"⚠️ {len(still)} rows still missing → timestamp fallback")
        df["ts_sec"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
        fb_train = df[df["consume_clean"].notna()]
        fb_pred  = df[df["consume_clean"].isna()]
        fb_pred  = fb_pred[fb_pred["ts_sec"].notna()].drop_duplicates(subset="timestamp")
        if not fb_train.empty and not fb_pred.empty:
            fb_model = LinearRegression().fit(fb_train[["ts_sec"]], fb_train["consume_clean"])
            y_fb = fb_model.predict(fb_pred[["ts_sec"]])
            df.loc[fb_pred.index, "consume_clean"] = pd.Series(y_fb, index=fb_pred.index)
        df.drop(columns=["ts_sec"], inplace=True)
    # Giá trị cuối và thải giá trị thừa
    df["consume"] = df["consume_clean"]
    # Đánh dấu những bản ghi vẫn còn thiếu consume
    # Khi hàm trả về, mỗi dòng sẽ có need_backfill = True/False.
    df.loc[:, "need_backfill"] = df["consume"].isna()
    return df.drop(columns=["consume_clean"])

# ───────────── MQTT CALLBACKS ─────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("✅ MQTT connected")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"❌ MQTT connect failed: {rc}")

# Pipe chính và debug
def on_message(client, userdata, msg):
    ts = datetime.utcnow().isoformat()
    payload = msg.payload.decode(errors="replace")
    with open(RAW_CHECKPOINT_PATH,"a",encoding="utf-8") as f:
        f.write(f"{ts},{msg.topic},\"{payload}\"\n")
    row = parse_row(ts,msg.topic,payload)
    if row is None: return
    # Ghép vào cửa sổ và fill ngay
    df_win = pd.DataFrame(window)
    df_new = pd.concat([df_win, pd.DataFrame([row])], ignore_index=True)
    df_filled = fill_missing(df_new.tail(2))  # chỉ cần bản ghi trước & mới
    row_clean = df_filled.tail(1).to_dict("records")[0]
    row_clean["need_backfill"] = pd.isna(row_clean["consume"])
    # Gắn giá trị clean vào window session
    window.append(row_clean)
    upsert_mongo([row_clean])
    logger.info(f"📥 Stored row {row_clean['timestamp']}")

# ───────────── BACK-FILL WORKER ─────────────
def backfill_worker():
    while not stop_event.is_set():
        time.sleep(BACKFILL_INTERVAL)
        df_win = pd.DataFrame(window)
        pending_mask = df_win["need_backfill"]
        if not pending_mask.any():
            continue
        idxs = df_win[pending_mask].index
        cols = ["voltage", "current", "power"]
        imputer = KNNImputer(n_neighbors=3)
        df_win[cols] = imputer.fit_transform(df_win[cols])
        train = df_win[~pending_mask]
        if train.empty:
            continue
        model = LinearRegression().fit(train[cols], train["consume"])
        df_win.loc[idxs, "consume"] = model.predict(df_win.loc[idxs, cols])
        df_win.loc[idxs, "need_backfill"] = False
        # update deque
        for i in idxs:
            window[i].update(df_win.loc[i].to_dict())
        # Upload and merge current on Mongo
        upsert_mongo([window[i] for i in idxs])
        logger.info(f"🔄 Back-filled {len(idxs)} rows")

# ─────── FASTAPI ENDPOINTS ───────
@app.get("/fetch")
def fetch(Password: str):
    if Password != FETCH_PASS:
        raise HTTPException(status_code=401)
    logger.info("✅ Fetch request received")
    client = MongoClient(MONGO_URI)
    data = list(client[MONGO_DB][MONGO_COL].find({}, {"_id": 0}))
    return JSONResponse(data)

@app.get("/delete")
def delete(Password: str):
    if Password != FETCH_PASS:
        raise HTTPException(status_code=401)
    client = MongoClient(MONGO_URI)
    count = client[MONGO_DB][MONGO_COL].delete_many({}).deleted_count
    logger.info("⚠️ Delete request received")
    return {"message": f"🧨 Deleted {count} rows from MongoDB."}

@app.get("/load")
def load(Password: str):
    if Password != FETCH_PASS:
        raise HTTPException(status_code=401)
    client = MongoClient(MONGO_URI)
    df = pd.DataFrame(list(client[MONGO_DB][MONGO_COL].find({}, {"_id": 0})))
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found.")
    logger.info("✅ Download request received")
    df.to_csv(EXPORT_CSV_PATH, index=False)
    return FileResponse(EXPORT_CSV_PATH, filename="poptech_cleaned_data.csv", media_type="text/csv")

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ─────── BOOTSTRAP ───────
def mqtt_main():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# Handle parallel threads
if __name__ == "__main__":
    # Set signal handlers in main thread
    def handle_exit(sig, _):
        logger.info("🛑 Shutdown signal received")
        stop_event.set()
    for s in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(s, handle_exit)
    # Handle data ingestion from MQTT broker, and backfiller
    threading.Thread(target=backfill_worker, daemon=True).start()   # quét back-fill 10s/lần
    threading.Thread(target=mqtt_main, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=7860)

