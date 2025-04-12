import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, minute, col, avg, count, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np

os.environ["JAVA_HOME"] = "C:\\Java\\jdk"
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

os.environ["PYSPARK_PYTHON"] = "C:\\Users\\mahima.LAPTOP-N07DQAHT\\Desktop\\penv\\Scripts\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\mahima.LAPTOP-N07DQAHT\\Desktop\\penv\\Scripts\\python.exe"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("NYC Taxi Data Analysis") \
    .getOrCreate()

file_path = "hdfs://localhost:9000/bigdata/taxi.csv"

df = spark.read.option("header", "true").csv(file_path)

df = df.withColumn("trip_distance", col("trip_distance").cast("double")) \
       .withColumn("fare_amount", col("fare_amount").cast("double")) \
       .withColumn("passenger_count", col("passenger_count").cast("integer")) \
       .withColumn("pickup_datetime", col("pickup_datetime").cast("timestamp")) \
       .withColumn("dropoff_datetime", col("dropoff_datetime").cast("timestamp"))

print("Sample Taxi Data:")
df.show(5, truncate=False)

df.createOrReplaceTempView("taxi_trips")

payment_distribution = spark.sql("""
    SELECT payment_type, COUNT(*) as count
    FROM taxi_trips
    GROUP BY payment_type
    ORDER BY count DESC
""")

print("Payment Type Distribution:")
payment_distribution.show()

avg_fare_by_passengers = spark.sql("""
    SELECT passenger_count, AVG(fare_amount) as avg_fare
    FROM taxi_trips
    GROUP BY passenger_count
    ORDER BY passenger_count
""")

print("Average Fare by Passenger Count:")
avg_fare_by_passengers.show()

distance_vs_fare = spark.sql("""
    SELECT trip_distance, fare_amount
    FROM taxi_trips
    ORDER BY trip_distance
""")

print("Trip Distance vs Fare Amount (Sample):")
distance_vs_fare.show(5)

busiest_hours = spark.sql("""
    SELECT HOUR(pickup_datetime) as hour, COUNT(*) as trip_count
    FROM taxi_trips
    GROUP BY HOUR(pickup_datetime)
    ORDER BY hour
""")

print("Trips by Hour of Day:")
busiest_hours.show()

avg_distance_by_hour = spark.sql("""
    SELECT HOUR(pickup_datetime) as hour, AVG(trip_distance) as avg_distance
    FROM taxi_trips
    GROUP BY HOUR(pickup_datetime)
    ORDER BY hour
""")

print("Average Trip Distance by Hour:")
avg_distance_by_hour.show()

output_dir = "taxi_analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\n=== LINEAR REGRESSION ANALYSIS ===")

regression_data = df.filter((df.trip_distance > 0) & 
                           (df.fare_amount > 0) & 
                           (df.fare_amount < 100) &  
                           (df.trip_distance < 30))  

assembler = VectorAssembler(inputCols=["trip_distance"], outputCol="features")
regression_df = assembler.transform(regression_data)

train_data, test_data = regression_df.randomSplit([0.7, 0.3], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="fare_amount", 
                     maxIter=10, regParam=0.1, elasticNetParam=0.8)
lr_model = lr.fit(train_data)

print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

predictions = lr_model.transform(test_data)
predictions.select("trip_distance", "fare_amount", "prediction").show(10)

evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"R-squared (R²): {r2}")

payment_df = payment_distribution.toPandas()
passengers_df = avg_fare_by_passengers.toPandas()
distance_fare_df = distance_vs_fare.toPandas()
hours_df = busiest_hours.toPandas()
distance_hour_df = avg_distance_by_hour.toPandas()
predictions_df = predictions.select("trip_distance", "fare_amount", "prediction").toPandas()

plt.figure(figsize=(10, 7))
plt.pie(payment_df['count'], labels=payment_df['payment_type'], autopct='%1.1f%%', startangle=90)
plt.title('Payment Type Distribution')
plt.axis('equal')
plt.savefig(f"{output_dir}/payment_distribution.png")
plt.close()

plt.figure(figsize=(10, 7))
sns.barplot(x='passenger_count', y='avg_fare', data=passengers_df)
plt.title('Average Fare by Passenger Count')
plt.xlabel('Number of Passengers')
plt.ylabel('Average Fare ($)')
plt.savefig(f"{output_dir}/avg_fare_by_passengers.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.scatter(distance_fare_df['trip_distance'], distance_fare_df['fare_amount'], alpha=0.5)
plt.title('Trip Distance vs Fare Amount')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True, linestyle='--', alpha=0.7)

z = np.polyfit(distance_fare_df['trip_distance'].astype(float), 
               distance_fare_df['fare_amount'].astype(float), 1)
p = np.poly1d(z)
plt.plot(distance_fare_df['trip_distance'].astype(float), 
         p(distance_fare_df['trip_distance'].astype(float)), 
         "r--", linewidth=2)

plt.savefig(f"{output_dir}/distance_vs_fare.png")
plt.close()

plt.figure(figsize=(12, 7))
plt.plot(hours_df['hour'], hours_df['trip_count'], marker='o', linestyle='-', linewidth=2)
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/trips_by_hour.png")
plt.close()

plt.figure(figsize=(12, 7))
plt.plot(distance_hour_df['hour'], distance_hour_df['avg_distance'], 
         marker='o', linestyle='-', linewidth=2, color='green')
plt.title('Average Trip Distance by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Distance (miles)')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/avg_distance_by_hour.png")
plt.close()

plt.figure(figsize=(12, 8))
plt.scatter(predictions_df['trip_distance'], predictions_df['fare_amount'], 
           alpha=0.5, label='Actual Fares', color='blue')
plt.scatter(predictions_df['trip_distance'], predictions_df['prediction'], 
           alpha=0.5, label='Predicted Fares', color='red')

z_actual = np.polyfit(predictions_df['trip_distance'], predictions_df['fare_amount'], 1)
p_actual = np.poly1d(z_actual)
plt.plot(predictions_df['trip_distance'], p_actual(predictions_df['trip_distance']), 
         "b--", linewidth=2, label='Actual Trend')

z_pred = np.polyfit(predictions_df['trip_distance'], predictions_df['prediction'], 1)
p_pred = np.poly1d(z_pred)
plt.plot(predictions_df['trip_distance'], p_pred(predictions_df['trip_distance']), 
         "r--", linewidth=2, label='Predicted Trend')

plt.title(f'Linear Regression: Trip Distance vs Fare Amount\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(f"{output_dir}/linear_regression_analysis.png")
plt.close()

plt.figure(figsize=(12, 8))
predictions_df['residuals'] = predictions_df['fare_amount'] - predictions_df['prediction']
plt.scatter(predictions_df['prediction'], predictions_df['residuals'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot: Predicted Fare vs Residuals')
plt.xlabel('Predicted Fare Amount ($)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/residual_plot.png")
plt.close()

plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
plt.pie(payment_df['count'], labels=payment_df['payment_type'], autopct='%1.1f%%', startangle=90)
plt.title('Payment Type Distribution')
plt.axis('equal')

plt.subplot(2, 3, 2)
sns.barplot(x='passenger_count', y='avg_fare', data=passengers_df, ax=plt.gca())
plt.title('Average Fare by Passenger Count')
plt.xlabel('Number of Passengers')
plt.ylabel('Average Fare ($)')

plt.subplot(2, 3, 3)
plt.plot(hours_df['hour'], hours_df['trip_count'], marker='o', linestyle='-', linewidth=2)
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.xticks(range(0, 24, 2))
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 3, 4)
plt.scatter(distance_fare_df['trip_distance'], distance_fare_df['fare_amount'], alpha=0.5)
plt.title('Trip Distance vs Fare Amount')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 3, 5)
plt.scatter(predictions_df['trip_distance'], predictions_df['fare_amount'], 
           alpha=0.5, label='Actual Fares', color='blue')
plt.scatter(predictions_df['trip_distance'], predictions_df['prediction'], 
           alpha=0.5, label='Predicted Fares', color='red')
plt.title(f'Linear Regression\nR² = {r2:.4f}')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 3, 6)
plt.scatter(predictions_df['prediction'], predictions_df['residuals'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Fare Amount ($)')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{output_dir}/dashboard_with_regression.png")
plt.close()

time_period_stats = spark.sql("""
    SELECT 
        CASE 
            WHEN HOUR(pickup_datetime) BETWEEN 6 AND 11 THEN 'Morning'
            WHEN HOUR(pickup_datetime) BETWEEN 12 AND 17 THEN 'Afternoon'
            WHEN HOUR(pickup_datetime) BETWEEN 18 AND 23 THEN 'Evening'
            ELSE 'Night'
        END as time_period,
        COUNT(*) as trip_count,
        AVG(trip_distance) as avg_distance,
        AVG(fare_amount) as avg_fare,
        AVG(passenger_count) as avg_passengers
    FROM taxi_trips
    GROUP BY 
        CASE 
            WHEN HOUR(pickup_datetime) BETWEEN 6 AND 11 THEN 'Morning'
            WHEN HOUR(pickup_datetime) BETWEEN 12 AND 17 THEN 'Afternoon'
            WHEN HOUR(pickup_datetime) BETWEEN 18 AND 23 THEN 'Evening'
            ELSE 'Night'
        END
    ORDER BY 
        CASE 
            WHEN time_period = 'Morning' THEN 1
            WHEN time_period = 'Afternoon' THEN 2
            WHEN time_period = 'Evening' THEN 3
            ELSE 4
        END
""")

print("Trip Statistics by Time Period:")
time_period_stats.show()

time_period_df = time_period_stats.toPandas()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].bar(time_period_df['time_period'], time_period_df['trip_count'], color='skyblue')
axes[0, 0].set_title('Trip Count by Time Period')
axes[0, 0].set_ylabel('Number of Trips')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

axes[0, 1].bar(time_period_df['time_period'], time_period_df['avg_distance'], color='lightgreen')
axes[0, 1].set_title('Average Distance by Time Period')
axes[0, 1].set_ylabel('Average Distance (miles)')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

axes[1, 0].bar(time_period_df['time_period'], time_period_df['avg_fare'], color='salmon')
axes[1, 0].set_title('Average Fare by Time Period')
axes[1, 0].set_ylabel('Average Fare ($)')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1, 1].bar(time_period_df['time_period'], time_period_df['avg_passengers'], color='purple')
axes[1, 1].set_title('Average Passengers by Time Period')
axes[1, 1].set_ylabel('Average Passengers')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{output_dir}/time_period_analysis.png")
plt.close()

print(f"\nAnalysis complete! Visualizations saved to {output_dir}/")
print("Analyses performed:")
print("1. Payment Type Distribution")
print("2. Average Fare by Passenger Count")
print("3. Trip Distance vs Fare Amount")
print("4. Trips by Hour of Day")
print("5. Average Trip Distance by Hour")
print("6. Combined Dashboard with Linear Regression")
print("7. Trip Statistics by Time Period")
print("8. Linear Regression Analysis with Residual Plot")

spark.stop()
