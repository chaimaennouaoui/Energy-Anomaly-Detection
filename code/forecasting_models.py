from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

# -------------------------
# 1) Start Spark
# -------------------------
spark = SparkSession.builder \
    .appName("EnerGuard-Forecasting-ML") \
    .getOrCreate()

# -------------------------
# 2) Load ML-ready parquet
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "parquet", "energy_ml_ready")

df = spark.read.parquet(DATA_PATH)
print("✅ Loaded parquet:", df.count(), "rows")

# -------------------------
# 3) Features + label
# -------------------------
label_col = "Global_active_power"

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

# keep only required columns
df = df.select(feature_cols + [label_col]).dropna()

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", col(label_col).alias("label"))

# -------------------------
# 4) Train/Test Split
# -------------------------
train, test = data.randomSplit([0.8, 0.2], seed=42)
print("✅ Train:", train.count(), "| Test:", test.count())

# -------------------------
# 5) Evaluation function
# -------------------------
def evaluate_predictions(predictions, model_name):
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print(f"\n📌 Results for {model_name}:")
    print("✅ RMSE =", rmse)
    print("✅ MAE  =", mae)
    print("✅ R2   =", r2)

# -------------------------
# 6) Model 1: Linear Regression
# -------------------------
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)
lr_pred = lr_model.transform(test)

evaluate_predictions(lr_pred, "LinearRegression")

# Save model
lr_path = os.path.join(PROJECT_ROOT, "models", "forecast_model_lr")
lr_model.write().overwrite().save(lr_path)
print("✅ LinearRegression model saved to:", lr_path)

# -------------------------
# 7) Model 2: RandomForestRegressor
# -------------------------
rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10, seed=42)
rf_model = rf.fit(train)
rf_pred = rf_model.transform(test)

evaluate_predictions(rf_pred, "RandomForestRegressor")

# Save model
rf_path = os.path.join(PROJECT_ROOT, "models", "forecast_model_rf")
rf_model.write().overwrite().save(rf_path)
print("✅ RandomForest model saved to:", rf_path)

spark.stop()
