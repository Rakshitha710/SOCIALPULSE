from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round

spark = SparkSession.builder \
    .appName("SocialPulse") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n===== SOCIALPULSE BIG DATA ANALYSIS =====")

df = spark.read.csv(
    "dataset/twitter_training_cleaned.csv",
    header=True,
    inferSchema=True
)

print("\nDataset Schema:")
df.printSchema()

print("\nTotal Records:")
print(df.count())

print("\n===== SENTIMENT DISTRIBUTION =====")

sentiment_counts = df.groupBy("Sentiment") \
    .agg(count("*").alias("Total_Tweets"))

sentiment_counts.show()

print("\n===== SENTIMENT PERCENTAGE =====")

total = df.count()

sentiment_percentage = sentiment_counts.withColumn(
    "Percentage",
    round((col("Total_Tweets") / total) * 100, 2)
)

sentiment_percentage.show()

print("\n===== TOP 10 TOPICS =====")

topic_counts = df.groupBy("Topic") \
    .agg(count("*").alias("Total_Tweets")) \
    .orderBy(col("Total_Tweets").desc())

topic_counts.show(10)

print("\n===== TOPIC-WISE SENTIMENT ANALYSIS =====")

topic_sentiment = df.groupBy("Topic", "Sentiment") \
    .agg(count("*").alias("Total_Tweets")) \
    .orderBy("Topic")

topic_sentiment.show(30)

spark.stop()