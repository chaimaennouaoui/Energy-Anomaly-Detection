from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os, shutil

spark = SparkSession.builder.appName("EnerGuard-Generate-Stream-Batches-MLReady").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ On utilise energy_ml_ready (même dataset que training)
PARQUET_PATH = os.path.join(PROJECT_ROOT, "output", "parquet", "energy_ml_ready")
ARCHIVE_DIR  = os.path.join(PROJECT_ROOT, "streaming", "archive")
INCOMING_DIR = os.path.join(PROJECT_ROOT, "streaming", "incoming")

os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(INCOMING_DIR, exist_ok=True)

# vider incoming et archive
for folder in [INCOMING_DIR, ARCHIVE_DIR]:
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp):
            os.remove(fp)

print("✅ incoming et archive vidés.")

df = spark.read.parquet(PARQUET_PATH) \
    .select(
        "timestamp",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "hour",
        "dayofweek",
        "month",
        "is_weekend"
    ) \
    .orderBy(col("timestamp"))

total = df.count()
print("✅ parquet ML-ready loaded:", total)

BATCH_SIZE = 5000
MAX_BATCHES = 50
num_batches = min(MAX_BATCHES, (total // BATCH_SIZE) + 1)

print(f"✅ Generating {num_batches} batches...")

for i in range(num_batches):
    start = i * BATCH_SIZE
    batch_df = df.limit(start + BATCH_SIZE).subtract(df.limit(start))

    batch_path = os.path.join(ARCHIVE_DIR, f"batch_{i+1:04d}.csv")

    batch_df.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(batch_path + "_tmp")

    tmp_folder = batch_path + "_tmp"
    part_file = [f for f in os.listdir(tmp_folder) if f.startswith("part-")][0]

    shutil.move(os.path.join(tmp_folder, part_file), batch_path)
    shutil.rmtree(tmp_folder)

    print(f"✅ batch {i+1}/{num_batches} -> {batch_path}")

print("\n🎉 Batches saved in:", ARCHIVE_DIR)
spark.stop()
