from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, when, round, concat, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import os, json

# -------------------------------------------------------
# 1) Spark
# -------------------------------------------------------
spark = SparkSession.builder.appName("EnerGuard-Streaming-Forecasting-LR").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INCOMING_DIR = os.path.join(PROJECT_ROOT, "streaming", "incoming")
PRED_DIR     = os.path.join(PROJECT_ROOT, "streaming", "predictions")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "forecast_model_lr")
THRESH_FILE  = os.path.join(PROJECT_ROOT, "output", "thresholds.json")

os.makedirs(PRED_DIR, exist_ok=True)

# -------------------------------------------------------
# 2) Load threshold
# -------------------------------------------------------
with open(THRESH_FILE, "r") as f:
    threshold = json.load(f)["threshold"]

print("✅ Threshold loaded:", threshold)

# -------------------------------------------------------
# 3) Load LR model
# -------------------------------------------------------
lr_model = LinearRegressionModel.load(MODEL_PATH)
print("✅ LR model loaded:", MODEL_PATH)

# -------------------------------------------------------
# 4) Streaming schema (ML-ready batches)
# -------------------------------------------------------
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("Global_active_power", DoubleType(), True),
    StructField("Global_reactive_power", DoubleType(), True),
    StructField("Voltage", DoubleType(), True),
    StructField("Global_intensity", DoubleType(), True),
    StructField("Sub_metering_1", DoubleType(), True),
    StructField("Sub_metering_2", DoubleType(), True),
    StructField("Sub_metering_3", DoubleType(), True),
    StructField("hour", IntegerType(), True),
    StructField("dayofweek", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("is_weekend", IntegerType(), True),
])

df = spark.readStream.option("header", True).schema(schema).csv(INCOMING_DIR)

# -------------------------------------------------------
# 5) Assemble features (same as training)
# -------------------------------------------------------
feature_cols = [
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
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_feat = assembler.transform(df)

# -------------------------------------------------------
# 6) Predict
# -------------------------------------------------------
pred = lr_model.transform(df_feat)

# anomalies
pred = pred.withColumn("anom_thr", when(col("Global_active_power") > threshold, 1).otherwise(0))
pred = pred.withColumn("abs_error", abs(col("Global_active_power") - col("prediction")))
pred = pred.withColumn("anom_err", when(col("abs_error") > 1.5, 1).otherwise(0))

# -------------------------------------------------------
# 7) Cleaner Display (short names + rounding)
# -------------------------------------------------------
display_df = pred.select(
    "timestamp",
    round(col("Global_active_power"), 3).alias("real"),
    round(col("prediction"), 3).alias("pred"),
    round(col("abs_error"), 3).alias("error"),
    col("anom_thr"),
    col("anom_err"),
    when((col("anom_thr") == 1) | (col("anom_err") == 1),
         concat(lit("🚨 ALERT: anomaly detected (real="),
                round(col("Global_active_power"), 3),
                lit(")"))
    ).otherwise(lit("✅ OK")).alias("status")
)

# -------------------------------------------------------
# ✅ OPTION: show only anomalies in console
# -------------------------------------------------------
SHOW_ONLY_ANOMALIES = False  # change to True if you want only anomalies

if SHOW_ONLY_ANOMALIES:
    display_console = display_df.filter((col("anom_thr") == 1) | (col("anom_err") == 1))
else:
    display_console = display_df

# -------------------------------------------------------
# 8) Streaming Output (Console + CSV)
# -------------------------------------------------------
query_console = display_console.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .option("numRows", 20) \
    .start()

query_files = display_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("header", True) \
    .option("path", PRED_DIR) \
    .option("checkpointLocation", os.path.join(PRED_DIR, "_checkpoint")) \
    .start()

print("\n✅ Streaming started successfully ✅")
print("📌 Incoming folder:", INCOMING_DIR)
print("📌 Predictions saved to:", PRED_DIR)
print("⚠️ Stop streaming with CTRL + C\n")

query_console.awaitTermination()
