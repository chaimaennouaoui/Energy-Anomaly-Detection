from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, concat_ws, hour, dayofmonth, month, dayofweek,
    when, avg, min as spark_min, max as spark_max, stddev
)
import os

# -------------------------------------------------------
# 1) Spark Session
# -------------------------------------------------------
spark = SparkSession.builder \
    .appName("EnerGuard-Preprocessing-Analysis") \
    .getOrCreate()

# -------------------------------------------------------
# 2) Paths (structure EnerGuard)
# -------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_FILE = os.path.join(PROJECT_ROOT, "data", "household_power_consumption.txt")
OUT_PARQUET = os.path.join(PROJECT_ROOT, "output", "parquet", "energy_cleaned")
OUT_AVG_HOUR = os.path.join(PROJECT_ROOT, "output", "parquet", "avg_hour")
OUT_AVG_DAY = os.path.join(PROJECT_ROOT, "output", "parquet", "avg_day")

print("📌 RAW FILE:", RAW_FILE)

# -------------------------------------------------------
# 3) Load dataset UCI (separator = ;)
# -------------------------------------------------------
df = spark.read.option("header", "true") \
    .option("sep", ";") \
    .option("inferSchema", "false") \
    .csv(RAW_FILE)

print("✅ Raw loaded. Rows:", df.count(), "| Cols:", len(df.columns))

# -------------------------------------------------------
# 4) Create timestamp from Date + Time
# -------------------------------------------------------
df = df.withColumn(
    "timestamp",
    to_timestamp(concat_ws(" ", col("Date"), col("Time")), "d/M/yyyy H:mm:ss")
)

df = df.drop("Date", "Time")

# -------------------------------------------------------
# 5) Convert numeric columns + handle '?' values
# -------------------------------------------------------
numeric_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]

for c in numeric_cols:
    df = df.withColumn(c, when(col(c) == "?", None).otherwise(col(c)).cast("double"))

# Drop rows missing essential fields
df = df.dropna(subset=["timestamp", "Global_active_power"])
df = df.dropna(subset=numeric_cols)

print("✅ After cleaning nulls. Rows:", df.count())

# -------------------------------------------------------
# 6) Feature Engineering
# -------------------------------------------------------
df = df.withColumn("hour", hour(col("timestamp"))) \
       .withColumn("day", dayofmonth(col("timestamp"))) \
       .withColumn("month", month(col("timestamp"))) \
       .withColumn("weekday", dayofweek(col("timestamp"))) \
       .withColumn("is_weekend", when(col("weekday").isin(1, 7), 1).otherwise(0))

print("✅ Time features created (hour, day, month, weekday, is_weekend)")

df.select("timestamp", "Global_active_power", "Voltage", "hour", "weekday", "is_weekend").show(5, False)

# -------------------------------------------------------
# 7) Exploratory analysis : AVG consumption by hour & day
# -------------------------------------------------------
avg_hour = df.groupBy("hour").agg(avg("Global_active_power").alias("avg_consumption_hour"))
avg_day = df.groupBy("day").agg(avg("Global_active_power").alias("avg_consumption_day"))

print("\n📌 Average consumption per hour:")
avg_hour.orderBy("hour").show(24)

print("\n📌 Average consumption per day:")
avg_day.orderBy("day").show(31)

# -------------------------------------------------------
# 8) Spark SQL analysis
# -------------------------------------------------------
df.createOrReplaceTempView("energy")

stats = spark.sql("""
SELECT
    MIN(Global_active_power) as min_consumption,
    MAX(Global_active_power) as max_consumption,
    AVG(Global_active_power) as mean_consumption,
    STDDEV(Global_active_power) as std_consumption
FROM energy
""")

print("\n📌 Descriptive statistics (SQL):")
stats.show()

top_peaks = spark.sql("""
SELECT timestamp, Global_active_power, hour, weekday, is_weekend
FROM energy
ORDER BY Global_active_power DESC
LIMIT 10
""")

print("\n📌 Top 10 peaks (highest consumption):")
top_peaks.show(10, False)

# -------------------------------------------------------
# 9) Save outputs in Parquet
# -------------------------------------------------------
df.write.mode("overwrite").parquet(OUT_PARQUET)
avg_hour.write.mode("overwrite").parquet(OUT_AVG_HOUR)
avg_day.write.mode("overwrite").parquet(OUT_AVG_DAY)

print("\n✅ Preprocessing + Analysis done ✅")
print("✅ Clean data saved to:", OUT_PARQUET)
print("✅ avg_hour saved to:", OUT_AVG_HOUR)
print("✅ avg_day saved to:", OUT_AVG_DAY)

spark.stop()
