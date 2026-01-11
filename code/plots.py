from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, concat_ws, hour, dayofmonth, month, dayofweek,
    when, avg
)
import os
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------
# 1) Spark Session
# -------------------------------------------------------
spark = SparkSession.builder \
    .appName("EnerGuard-Preprocessing-Exploration") \
    .getOrCreate()

# -------------------------------------------------------
# 2) Paths
# -------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_FILE = os.path.join(PROJECT_ROOT, "data", "household_power_consumption.txt")

OUT_PARQUET_CLEAN = os.path.join(PROJECT_ROOT, "output", "parquet", "energy_cleaned")
OUT_AVG_HOUR = os.path.join(PROJECT_ROOT, "output", "parquet", "avg_hour")
OUT_AVG_DAY = os.path.join(PROJECT_ROOT, "output", "parquet", "avg_day")

OUTPUT_GRAPHS = os.path.join(PROJECT_ROOT, "output", "graphs")
os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

# -------------------------------------------------------
# 3) Load dataset
# -------------------------------------------------------
df = spark.read.option("header", "true") \
    .option("sep", ";") \
    .option("inferSchema", "false") \
    .csv(RAW_FILE)

print("✅ Raw dataset loaded. Rows:", df.count(), "| Cols:", len(df.columns))

# -------------------------------------------------------
# 4) Clean & convert
# -------------------------------------------------------
df = df.withColumn(
    "timestamp",
    to_timestamp(concat_ws(" ", col("Date"), col("Time")), "d/M/yyyy H:mm:ss")
).drop("Date", "Time")

numeric_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]

for c in numeric_cols:
    df = df.withColumn(c, when(col(c)=="?", None).otherwise(col(c)).cast("double"))

df = df.dropna(subset=["timestamp"] + numeric_cols)

# Feature engineering
df = df.withColumn("hour", hour(col("timestamp"))) \
       .withColumn("day", dayofmonth(col("timestamp"))) \
       .withColumn("month", month(col("timestamp"))) \
       .withColumn("weekday", dayofweek(col("timestamp"))) \
       .withColumn("is_weekend", when(col("weekday").isin(1,7), 1).otherwise(0))

# Optional: remove extreme peaks > 12 kW
df = df.filter(col("Global_active_power") <= 12)

print("✅ Preprocessing done. Rows after cleaning:", df.count())

# Save cleaned dataset
df.write.mode("overwrite").parquet(OUT_PARQUET_CLEAN)

# -------------------------------------------------------
# 5) Exploratory analysis
# -------------------------------------------------------
df.createOrReplaceTempView("energy")

print("\n📌 SQL stats:")
spark.sql("""
SELECT
    MIN(Global_active_power) AS min_power,
    MAX(Global_active_power) AS max_power,
    AVG(Global_active_power) AS mean_power,
    STDDEV(Global_active_power) AS std_power
FROM energy
""").show()

print("\n📌 Top 10 peaks:")
spark.sql("""
SELECT timestamp, Global_active_power, hour, weekday, is_weekend
FROM energy
ORDER BY Global_active_power DESC
LIMIT 10
""").show(truncate=False)

# Average consumption per hour
avg_hour = df.groupBy("hour").agg(avg("Global_active_power").alias("avg_consumption_hour"))
avg_hour.write.mode("overwrite").parquet(OUT_AVG_HOUR)

# Average consumption per day
avg_day = df.groupBy("day").agg(avg("Global_active_power").alias("avg_consumption_day"))
avg_day.write.mode("overwrite").parquet(OUT_AVG_DAY)

avg_weekday = df.groupBy("weekday").agg(avg("Global_active_power").alias("avg_consumption_weekday"))

# -------------------------------------------------------
# 6) Correlation (SAFE sample)
# -------------------------------------------------------
df_sample_pd = df.select(
    "Global_active_power","Voltage","Global_reactive_power","Global_intensity"
).sample(False, 0.01, seed=42).toPandas()

corr_matrix = df_sample_pd.corr()
print("\n📌 Correlation matrix (sample 1%):")
print(corr_matrix)

# Save correlation matrix to csv
corr_matrix.to_csv(os.path.join(OUTPUT_GRAPHS, "correlation_matrix.csv"))

# -------------------------------------------------------
# 7) Graphs
# -------------------------------------------------------

# Avg per hour
avg_hour_pd = avg_hour.orderBy("hour").toPandas()
plt.figure()
plt.plot(avg_hour_pd['hour'], avg_hour_pd['avg_consumption_hour'], marker='o')
plt.title("Consommation moyenne par heure")
plt.xlabel("Heure")
plt.ylabel("Consommation (kW)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_GRAPHS, "avg_consumption_hour.png"))
plt.close()

# Avg per weekday
avg_weekday_pd = avg_weekday.orderBy("weekday").toPandas()
plt.figure()
plt.bar(avg_weekday_pd['weekday'], avg_weekday_pd['avg_consumption_weekday'])
plt.title("Consommation moyenne par jour de la semaine")
plt.xlabel("Jour (1=Dimanche, 7=Samedi)")
plt.ylabel("Consommation (kW)")
plt.grid(axis='y')
plt.savefig(os.path.join(OUTPUT_GRAPHS, "avg_consumption_weekday.png"))
plt.close()

# Scatter consommation vs voltage (sample)
plt.figure()
plt.scatter(df_sample_pd['Voltage'], df_sample_pd['Global_active_power'], alpha=0.3)
plt.title("Consommation vs Voltage (sample)")
plt.xlabel("Voltage (V)")
plt.ylabel("Global_active_power (kW)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_GRAPHS, "scatter_consumption_voltage.png"))
plt.close()

# Histogramme consommation (sample)
plt.figure()
plt.hist(df_sample_pd['Global_active_power'], bins=50)
plt.title("Distribution Global_active_power (sample)")
plt.xlabel("Global_active_power (kW)")
plt.ylabel("Fréquence")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_GRAPHS, "hist_consumption.png"))
plt.close()

print("\n✅ Graphs saved in:", OUTPUT_GRAPHS)
spark.stop()
