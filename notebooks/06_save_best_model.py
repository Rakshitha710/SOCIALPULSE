import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

project_path = Path(__file__).resolve().parent.parent

train = pd.read_csv(project_path / "dataset" / "twitter_training_cleaned.csv")
validation = pd.read_csv(project_path / "dataset" / "twitter_validation_cleaned.csv")

train = train.dropna(subset=["Clean_Tweet", "Sentiment"])
validation = validation.dropna(subset=["Clean_Tweet", "Sentiment"])

X_train = train["Clean_Tweet"]
y_train = train["Sentiment"]

X_test = validation["Clean_Tweet"]
y_test = validation["Sentiment"]

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LinearSVC(class_weight="balanced")

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

print("Linear SVM Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

models_path = project_path / "models"
models_path.mkdir(exist_ok=True)

joblib.dump(model, models_path / "best_sentiment_model.pkl")
joblib.dump(tfidf, models_path / "best_tfidf_vectorizer.pkl")

print("\nBest model saved successfully!")