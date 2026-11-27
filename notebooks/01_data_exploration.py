import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

train_path = project_path / "dataset" / "twitter_training.csv"
validation_path = project_path / "dataset" / "twitter_validation.csv"

columns = ["Tweet_ID", "Topic", "Sentiment", "Tweet"]

train = pd.read_csv(train_path, header=None, names=columns)
validation = pd.read_csv(validation_path, header=None, names=columns)

print("TRAINING DATA")
print(train.head())

print("\nVALIDATION DATA")
print(validation.head())

print("\nTRAINING SHAPE:", train.shape)
print("VALIDATION SHAPE:", validation.shape)

print("\nSENTIMENT DISTRIBUTION")
print(train["Sentiment"].value_counts())

print("\nTOP 10 TOPICS")
print(train["Topic"].value_counts().head(10))

print("\nMISSING VALUES")
print(train.isnull().sum())