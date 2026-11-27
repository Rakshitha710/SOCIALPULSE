import joblib
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

model = joblib.load(project_path / "models" / "best_sentiment_model.pkl")

tfidf = joblib.load(project_path / "models" / "best_tfidf_vectorizer.pkl")

tweets = [
    "This product is absolutely amazing and I love it",
    "The worst experience ever, completely disappointed",
    "The product was released yesterday",
    "I have no interest in this topic"
]

tweets_tfidf = tfidf.transform(tweets)

predictions = model.predict(tweets_tfidf)

print("===== SENTIMENT PREDICTIONS =====")

for tweet, sentiment in zip(tweets, predictions):
    print("\nTweet:", tweet)
    print("Predicted Sentiment:", sentiment)