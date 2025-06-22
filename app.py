# app.py
# Root API: https://binkhoale1812-poptech-cleaner.hf.space/
# Usages: 
## https://binkhoale1812-poptech-cleaner.hf.space/fetch?Password=...
## https://binkhoale1812-poptech-cleaner.hf.space/load?Password=...
## https://binkhoale1812-poptech-cleaner.hf.space/delete?Password=...

import os, json, signal, logging, threading, time
from datetime import datetime, timedelta
from queue import Queue, Empty

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

BATCH_SECONDS = int(os.getenv("WINDOW_SECONDS", 1800))
EXPECTED_INTERVAL_SEC = int(os.getenv("EXPECTED_INTERVAL_SEC", 30))
TOLERANCE_SEC = int(os.getenv("TOLERANCE_SEC", 10))
RAW_CHECKPOINT_PATH = os.getenv("RAW_CHECKPOINT_PATH", "cache/checkpoint_raw.csv")
EXPORT_CSV_PATH = "mongo_cleaned_export.csv"

os.makedirs(os.path.dirname(RAW_CHECKPOINT_PATH), exist_ok=True)

# ─────── LOGGING ───────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    force=True
)
logger = logging.getLogger("poptech-cleaner")
for m in ["pymongo", "pymongo.server_selection", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(m).setLevel(logging.WARNING)
logger.info("🚀 PopTech FastAPI Cleaning Server starting...")

# ──────────── GLOBALS ─────────────────
queue_raw = Queue()
stop_event = threading.Event()
app = FastAPI()

# ─────────── MQTT CALLBACKS ───────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"❌ MQTT connection failed: {rc}")

# ─ DEBUG MESSENGER & CHECKPOINT WRITER ─
def on_message(client, userdata, msg):
    ts = datetime.utcnow().isoformat()
    payload = msg.payload.decode(errors="replace")
    queue_raw.put({"timestamp": ts, "topic": msg.topic, "payload": payload})
    # Clean out spaces
    try:
        data = json.loads(payload.replace('""', '"')).get("data", [])
        logger.info(f"📩 MQTT: {ts} | V={data[0] if len(data)>0 else None}V, A={data[1] if len(data)>1 else None}A, W={data[2] if len(data)>2 else None}W, mWh={data[3] if len(data)>3 else None}")
    except Exception:
        pass
    # Return as compact of ts, topic and payload at this stage
    try:
        with open(RAW_CHECKPOINT_PATH, "a", encoding="utf-8") as f:
            f.write(f'{ts},{msg.topic},"{payload}"\n')
    except Exception as e:
        logger.error(f"❌ Failed to write checkpoint: {e}")

# ───────────── PIPELINE ─────────────
## Filter and parsing payload to 4 individual variables
def parse_and_filter(raw_rows):
    rows = []
    for r in raw_rows:
        try:
            payload = json.loads(r["payload"].replace('""', '"'))
            if r["topic"].startswith("device/socket/reply/") and isinstance(payload.get("data", []), list):
                v, a, w, c = (payload["data"] + [None]*4)[:4]
                if any(x not in (0, None) for x in (a, w, c)):
                    rows.append({
                        "timestamp": r["timestamp"],
                        "id": payload.get("id"),
                        "imei": payload.get("imei"),
                        "type": payload.get("type"),
                        "voltage": float(v),
                        "current": float(a),
                        "power": float(w),
                        "consume": float(c),
                    })
        except:
            continue
    return pd.DataFrame(rows)

## Detect and fill missing
def fill_missing(df):
    if df.empty:
        return df
    # --- Chuẩn hoá thời gian ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    # Time interval limitation
    expected = timedelta(seconds=EXPECTED_INTERVAL_SEC)
    tol = timedelta(seconds=TOLERANCE_SEC)
    # --- B1. Chèn bản ghi bị rơi ---
    rows = [df.iloc[0]]
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1]["timestamp"], df.iloc[i]["timestamp"]
        rows.append(df.iloc[i])
        if curr - prev > expected + tol:
            for j in range(1, int(round((curr - prev) / expected))):
                gap_ts = prev + j * expected
                gap = df.iloc[i - 1].copy()
                gap["timestamp"] = gap_ts
                for col in ["voltage", "current", "power", "consume"]:
                    gap[col] = np.nan
                rows.insert(-1, gap)
    # Sorting with ts to be identifier
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["consume_clean"] = df["consume"]
    df.loc[(df["consume"] < 0) | (df["consume"].diff() < 0), "consume_clean"] = np.nan
    # --- B2. Impute feature ---
    imputer = KNNImputer(n_neighbors=3)
    df[["voltage", "current", "power"]] = imputer.fit_transform(
        df[["voltage", "current", "power"]]
    )
    # --- B3. Model chính ---
    train = df[df["consume_clean"].notna()]
    pred  = df[df["consume_clean"].isna()]
    # NaN and null not valid
    if not train.empty and not pred.empty:
        model = LinearRegression().fit(
            train[["voltage", "current", "power"]],
            train["consume_clean"]
        )
        try:
            y_hat = model.predict(pred[["voltage", "current", "power"]])
            # Khớp index bằng Series (an toàn với duplicate)
            df.loc[pred.index, "consume_clean"] = pd.Series(y_hat, index=pred.index)
        except Exception as e:
            logger.warning(f"⚠️ Primary model failed partially: {e}")
    # --- B4. Fallback theo timestamp ---
    still_missing = df[df["consume_clean"].isna()]
    if not still_missing.empty:
        logger.warning(f"⚠️ {len(still_missing)} rows still missing. Using timestamp fallback.")
        df["ts_sec"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
        fb_train = df[df["consume_clean"].notna()]
        fb_pred  = df[df["consume_clean"].isna()]
        # Chỉ lấy bản ghi có ts_sec hợp lệ & index duy nhất
        fb_pred = fb_pred[fb_pred["ts_sec"].notna()].drop_duplicates(subset="timestamp")
        if not fb_train.empty and not fb_pred.empty:
            fb_model = LinearRegression().fit(
                fb_train[["ts_sec"]], fb_train["consume_clean"]
            )
            y_fb = fb_model.predict(fb_pred[["ts_sec"]])
            df.loc[fb_pred.index, "consume_clean"] = pd.Series(y_fb, index=fb_pred.index)
        # Drop total sec temp var
        df.drop(columns=["ts_sec"], inplace=True)
    # --- Kết quả cuối ---
    df["consume"] = df["consume_clean"]
    logger.info("🧹 fill_missing() hoàn tất làm sạch & khôi phục.")
    return df.drop(columns=["consume_clean"])

## MongoDB insertion 
def insert_mongo(df):
    if df.empty: return
    try:
        client = MongoClient(MONGO_URI)
        col = client[MONGO_DB][MONGO_COL]
        col.create_index("timestamp", unique=True)
        records = df.to_dict("records")
        for r in records: r["_id"] = r["timestamp"]
        operations = [
                    UpdateOne({"_id": r["_id"]}, {"$set": r}, upsert=True) for r in records
                ]
        col.bulk_write(operations, ordered=False)
        logger.info(f"🪣 Inserted {len(records)} rows to MongoDB.")
    except Exception as e:
        logger.error(f"❌ Mongo insert error: {e}")

## Batch worker to insert data to MongoDB
def batch_worker():
    while not stop_event.is_set():
        time.sleep(BATCH_SECONDS)
        bundle = []
        while True:
            try: bundle.append(queue_raw.get_nowait())
            except Empty: break
        if not bundle:
            logger.debug("⏱️ No new data this cycle")
            continue
        logger.info("Start cleaning 🧹")
        df_clean = fill_missing(parse_and_filter(bundle))
        insert_mongo(df_clean)

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
    # MQTT broker ingestion
    threading.Thread(target=batch_worker, daemon=True).start()
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
    # Handle data ingestion from MQTT broker
    threading.Thread(target=mqtt_main, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=7860)

