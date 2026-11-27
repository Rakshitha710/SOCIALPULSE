import pandas as pd
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

results = []

# EXPERIMENT 1: WORD-LEVEL TF-IDF

print("\nEXPERIMENT 1: WORD-LEVEL TF-IDF")

word_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

word_model = LinearSVC(class_weight="balanced")

word_model.fit(X_train_word, y_train)

word_predictions = word_model.predict(X_test_word)

word_accuracy = accuracy_score(y_test, word_predictions)

print("Accuracy:", word_accuracy)

print(classification_report(
    y_test,
    word_predictions,
    zero_division=0
))

results.append({
    "Model": "Word TF-IDF 20000",
    "Accuracy": word_accuracy,
    "Neutral Precision": classification_report(
        y_test, word_predictions, output_dict=True, zero_division=0
    )["Neutral"]["precision"],
    "Neutral Recall": classification_report(
        y_test, word_predictions, output_dict=True, zero_division=0
    )["Neutral"]["recall"]
})


# EXPERIMENT 2: WORD + CHARACTER TF-IDF

print("\nEXPERIMENT 2: WORD + CHARACTER TF-IDF")

from sklearn.pipeline import FeatureUnion

combined_vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=10000,
        sublinear_tf=True
    ))
])

X_train_combined = combined_vectorizer.fit_transform(X_train)
X_test_combined = combined_vectorizer.transform(X_test)

combined_model = LinearSVC(class_weight="balanced")

combined_model.fit(X_train_combined, y_train)

combined_predictions = combined_model.predict(X_test_combined)

combined_accuracy = accuracy_score(y_test, combined_predictions)

print("Accuracy:", combined_accuracy)

print(classification_report(
    y_test,
    combined_predictions,
    zero_division=0
))

combined_report = classification_report(
    y_test, combined_predictions, output_dict=True, zero_division=0
)

results.append({
    "Model": "Word + Character TF-IDF",
    "Accuracy": combined_accuracy,
    "Neutral Precision": combined_report["Neutral"]["precision"],
    "Neutral Recall": combined_report["Neutral"]["recall"]
})


# SAVE EXPERIMENT RESULTS

results_df = pd.DataFrame(results)

results_path = project_path / "dataset" / "experiment_results.csv"

results_df.to_csv(results_path, index=False)

print("\nMODEL COMPARISON RESULTS")
print(results_df.to_string(index=False))

print("\nExperiment completed successfully!")
print("Results saved to:", results_path)

import joblib

models_path = project_path / "models"
models_path.mkdir(exist_ok=True)

joblib.dump(
    combined_model,
    models_path / "best_sentiment_model.pkl"
)

joblib.dump(
    combined_vectorizer,
    models_path / "best_tfidf_vectorizer.pkl"
)

print("\nBest experimental model saved successfully!")