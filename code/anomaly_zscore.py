from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, when, desc
import os
import json

# -------------------------
# 1) Spark Session
# -------------------------
spark = SparkSession.builder \
    .appName("EnerGuard-Anomaly-Zscore") \
    .getOrCreate()

# -------------------------
# 2) Load ML-ready parquet
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "parquet", "energy_ml_ready")

df = spark.read.parquet(DATA_PATH).select("timestamp", "Global_active_power")
print("✅ Loaded parquet:", df.count(), "rows")

# -------------------------
# 3) Compute mean and std
# -------------------------
stats = df.select(
    mean("Global_active_power").alias("mean_power"),
    stddev("Global_active_power").alias("std_power")
).collect()[0]

mean_power = stats["mean_power"]
std_power = stats["std_power"]
threshold = mean_power + 3 * std_power

print("\n📌 Z-score Statistics:")
print("✅ Mean =", mean_power)
print("✅ Std  =", std_power)
print("✅ Threshold (mean + 3*std) =", threshold)

# -------------------------
# 4) Detect anomalies
# -------------------------
df_anom = df.withColumn(
    "is_anomaly",
    when(col("Global_active_power") > threshold, 1).otherwise(0)
)

anom_count = df_anom.filter(col("is_anomaly") == 1).count()
total = df_anom.count()

print("\n📌 Anomalies detected:")
print("✅ Total anomalies =", anom_count)
print("✅ Percentage =", (anom_count / total) * 100, "%")

print("\n📌 Top 20 anomalies (highest consumption):")
df_anom.filter(col("is_anomaly") == 1) \
    .orderBy(desc("Global_active_power")) \
    .show(20, False)

# -------------------------
# 5) Save thresholds for streaming (Person 3)
# -------------------------
out_json = os.path.join(PROJECT_ROOT, "output", "thresholds.json")

with open(out_json, "w") as f:
    json.dump({
        "mean_power": mean_power,
        "std_power": std_power,
        "threshold": threshold
    }, f, indent=4)

print("\n✅ thresholds.json saved to:", out_json)

spark.stop()
