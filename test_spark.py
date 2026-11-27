from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SocialPulse") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)

data = [("Positive", 100), ("Negative", 50), ("Neutral", 75)]

df = spark.createDataFrame(data, ["Sentiment", "Count"])

df.show()

spark.stop()