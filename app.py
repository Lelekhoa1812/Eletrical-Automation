import os
import json
import signal
import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue, Empty

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from pymongo import MongoClient, errors as mongo_errors

# ─────────────────────────  ENV / CONFIG  ──────────────────────────
load_dotenv()  # reads .env in container root

# Topic and APIs
BROKER      = os.getenv("BROKER")
PORT        = int(os.getenv("PORT", 1883))
USERNAME    = os.getenv("USERNAME")
PASSWORD    = os.getenv("PASSWORD")
MQTT_TOPIC  = os.getenv("MQTT_TOPIC", "device/socket/reply/#")

# Mongo string
MONGO_URI   = os.getenv("MONGO_URI")
MONGO_DB    = os.getenv("MONGO_DB", "poptech")
MONGO_COL   = os.getenv("MONGO_COLLECTION", "device_clean")

# Prediction and cleaning prefixes
BATCH_SECONDS          = int(os.getenv("WINDOW_SECONDS", 3600))      # 1 h default (suggesting ~15-30m on session saver on deployment stage)
EXPECTED_INTERVAL_SEC  = int(os.getenv("EXPECTED_INTERVAL_SEC", 30))
TOLERANCE_SEC          = int(os.getenv("TOLERANCE_SEC", 2))

# Write checkpoint file as cacheable
RAW_CHECKPOINT_PATH = os.getenv("RAW_CHECKPOINT_PATH", "cache/checkpoint_raw.csv")
os.makedirs(os.path.dirname(RAW_CHECKPOINT_PATH), exist_ok=True)

# ─────────────────────────  LOGGING  ───────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    force=True,
)
logger = logging.getLogger("poptech-cleaner")
for m in ["pymongo", "pymongo.server_selection",
          "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(m).setLevel(logging.WARNING)
logger.info("🚀 Starting PopTech Electrical Cleaning Pipeline...")

# ─────────────────────────  GLOBALS  ───────────────────────────────
queue_raw   = Queue()        # MQTT → queue → batch thread
stop_event  = threading.Event()


# ────────────────────────  MQTT CALLBACKS  ─────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC, qos=0)
    else:
        logger.error(f"❌ MQTT connection failed with code {rc}")


def on_message(client, userdata, msg):
    """Push raw line (timestamp, topic, payload) onto in-memory queue + CSV checkpoint"""
    ts = datetime.utcnow().isoformat()
    payload = msg.payload.decode(errors="replace")
    row_dict = {"timestamp": ts, "topic": msg.topic, "payload": payload}
    queue_raw.put(row_dict)

    # Log every received message (even before parsing)
    try:
        data = json.loads(payload.replace('""', '"')).get("data", [])
        voltage = data[0] if len(data) > 0 else None
        current = data[1] if len(data) > 1 else None
        power   = data[2] if len(data) > 2 else None
        consume = data[3] if len(data) > 3 else None
        logger.info(f"📩 MQTT received: timestamp: {ts}, voltage: {voltage}V, current: {current}A, power: {power}W, consume: {consume}mWh")
    except Exception as e:
        logger.warning(f"⚠️ Failed to parse MQTT message: {e} | payload: {payload}")

    # Append to cache CSV
    try:
        with open(RAW_CHECKPOINT_PATH, "a", encoding="utf-8") as f:
            f.write(f'{ts},{msg.topic},"{payload.replace(chr(34)*2, chr(34))}"\n')
    except Exception as e:
        logger.error(f"❌ Could not write checkpoint log: {e}")


# ─────────────────────────  CORE LOGIC  ────────────────────────────
def parse_and_filter(raw_rows):
    """
    raw_rows: list[dict] from MQTT queue
    returns: pd.DataFrame ready for cleaning
    """
    parsed_rows = []

    for r in raw_rows:
        try:
            payload = json.loads(r["payload"].replace('""', '"'))
            if not isinstance(payload, dict):
                continue
            # keep only valid socket rows
            if not r["topic"].startswith("device/socket/reply/"):
                continue

            data = payload.get("data", [])
            if not (isinstance(data, list) and len(data) >= 4):
                continue

            voltage, current, power, consume = data[:4]
            # drop idle frames where current, power, consume are 0 or None
            if all(v in (0, None) for v in (current, power, consume)):
                continue

            parsed_rows.append(
                {
                    "timestamp": r["timestamp"],
                    "id":        payload.get("id"),
                    "imei":      payload.get("imei"),
                    "type":      payload.get("type"),
                    "voltage":   float(voltage),
                    "current":   float(current),
                    "power":     float(power),
                    "consume":   float(consume),
                }
            )
        except Exception as e:
            logger.debug(f"⚠️ Skipping malformed payload: {e}")

    return pd.DataFrame(parsed_rows)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect >30 ± 2 s gaps; insert empty rows; impute/predict.
    """
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    expected = timedelta(seconds=EXPECTED_INTERVAL_SEC)
    tol      = timedelta(seconds=TOLERANCE_SEC)

    filled_rows   = [df.iloc[0]]
    missing_count = 0

    for i in range(1, len(df)):
        prev, cur = df.iloc[i - 1]["timestamp"], df.iloc[i]["timestamp"]
        delta     = cur - prev

        filled_rows.append(df.iloc[i])

        if delta > expected + tol:
            gaps = int(round(delta / expected)) - 1
            missing_count += gaps

            for j in range(1, gaps + 1):
                ts_gap = prev + j * expected
                newrow = df.iloc[i - 1].copy()
                newrow["timestamp"] = ts_gap
                for col in ["voltage", "current", "power", "consume"]:
                    newrow[col] = np.nan
                filled_rows.insert(-1, newrow)

    df_full = pd.DataFrame(filled_rows).sort_values("timestamp").reset_index(drop=True)

    # --- cleansing & model-based consume reconstruction -------------
    feature_cols  = ["voltage", "current", "power"]
    target_col    = "consume"

    df_full["consume_clean"] = df_full[target_col]
    diff = df_full["consume_clean"].diff()
    df_full.loc[(df_full["consume_clean"] < 0) | (diff < 0), "consume_clean"] = np.nan

    # KNN impute V, A, W
    imputer = KNNImputer(n_neighbors=3)
    X_raw   = df_full[feature_cols]
    X_filled= pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)
    df_full[feature_cols] = X_filled

    # predict missing consume via linear regression
    train = df_full[df_full["consume_clean"].notna()]
    pred  = df_full[df_full["consume_clean"].isna()]
    if not pred.empty and not train.empty:
        model = LinearRegression().fit(train[feature_cols], train["consume_clean"])
        df_full.loc[pred.index, "consume_clean"] = model.predict(df_full.loc[pred.index, feature_cols])

    df_full[target_col] = df_full["consume_clean"]
    df_full.drop(columns=["consume_clean"], inplace=True)

    logger.info(f"✨ Clean batch: {len(df)} ➜ {len(df_full)} rows "
                f"(filled {missing_count} gaps)")
    return df_full


def insert_mongo(df: pd.DataFrame):
    """
    Upsert cleaned docs into MongoDB.
    """
    if df.empty:
        return

    try:
        client     = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        collection = client[MONGO_DB][MONGO_COL]

        # ensure timestamp uniqueness to avoid duplicates
        collection.create_index("timestamp", unique=True)

        records = df.to_dict("records")
        for rec in records:
            rec["_id"] = rec["timestamp"]  # quick natural key
        collection.bulk_write(
            [pymongo.UpdateOne({"_id": r["_id"]}, {"$set": r}, upsert=True) for r in records],
            ordered=False,
        )
        logger.info(f"📥 MongoDB: upserted {len(records)} docs")
    except mongo_errors.BulkWriteError as bwe:
        logger.debug("Duplicate records skipped")
    except Exception as e:
        logger.error(f"❌ Mongo insert failed: {e}")


def batch_worker():
    """
    Every BATCH_SECONDS pull everything from queue, process, store.
    """
    while not stop_event.is_set():
        time.sleep(BATCH_SECONDS)

        bundle = []
        try:
            while True:
                bundle.append(queue_raw.get_nowait())
        except Empty:
            pass  # queue drained

        if not bundle:
            logger.debug("⏱️ Batch tick – no new data")
            continue

        df_parsed = parse_and_filter(bundle)
        df_clean  = fill_missing(df_parsed)
        insert_mongo(df_clean)


# ─────────────────────────  RUNTIME  ───────────────────────────────
def main():
    # start batch thread
    thr = threading.Thread(target=batch_worker, daemon=True)
    thr.start()

    # MQTT client in main thread
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)

    # graceful SIGTERM/SIGINT (HF Space shutdown)
    def handle_exit(signum, frame):
        logger.info("🛑 Shutdown signal received")
        stop_event.set()
        client.disconnect()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle_exit)

    # blocking loop
    client.loop_forever()


if __name__ == "__main__":
    main()
