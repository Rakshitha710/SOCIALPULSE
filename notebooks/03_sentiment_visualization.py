import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

data_path = project_path / "dataset" / "twitter_training_cleaned.csv"

df = pd.read_csv(data_path)

# Count each sentiment
sentiment_counts = df["Sentiment"].value_counts()

print("Sentiment Counts:")
print(sentiment_counts)

# Bar chart
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")

plt.title("Twitter Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(7, 7))
sentiment_counts.plot(kind="pie", autopct="%1.1f%%")

plt.title("Percentage of Each Sentiment")
plt.ylabel("")
plt.tight_layout()
plt.show()