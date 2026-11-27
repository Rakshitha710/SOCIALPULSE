import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

project_path = Path(__file__).resolve().parent.parent

train_path = project_path / "dataset" / "twitter_training_cleaned.csv"
validation_path = project_path / "dataset" / "twitter_validation_cleaned.csv"

train = pd.read_csv(train_path)
validation = pd.read_csv(validation_path)

# Remove rows with missing values
train = train.dropna(subset=["Clean_Tweet", "Sentiment"])
validation = validation.dropna(subset=["Clean_Tweet", "Sentiment"])

# Separate input and output
X_train = train["Clean_Tweet"]
y_train = train["Sentiment"]

X_validation = validation["Clean_Tweet"]
y_validation = validation["Sentiment"]

# Convert text into numerical features
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_validation_tfidf = tfidf.transform(X_validation)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict sentiments
predictions = model.predict(X_validation_tfidf)

# Evaluate model
print("MODEL ACCURACY:")
print(accuracy_score(y_validation, predictions))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_validation, predictions, zero_division=0))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_validation, predictions))

# Save model and vectorizer
models_path = project_path / "models"
models_path.mkdir(exist_ok=True)

joblib.dump(model, models_path / "sentiment_model.pkl")
joblib.dump(tfidf, models_path / "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")