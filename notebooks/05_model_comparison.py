import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1
    })

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    project_path / "dataset" / "model_comparison.csv",
    index=False
)

print("\nComparison results saved successfully!")