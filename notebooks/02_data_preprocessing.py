import pandas as pd
import re
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

train_path = project_path / "dataset" / "twitter_training.csv"
validation_path = project_path / "dataset" / "twitter_validation.csv"

columns = ["Tweet_ID", "Topic", "Sentiment", "Tweet"]

train = pd.read_csv(train_path, header=None, names=columns)
validation = pd.read_csv(validation_path, header=None, names=columns)

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Remove missing tweets
train = train.dropna(subset=["Tweet"]).copy()
validation = validation.dropna(subset=["Tweet"]).copy()

# Remove duplicate records
train = train.drop_duplicates()
validation = validation.drop_duplicates()

# Clean tweet text
train["Clean_Tweet"] = train["Tweet"].apply(clean_text)
validation["Clean_Tweet"] = validation["Tweet"].apply(clean_text)

# Remove empty tweets after cleaning
train = train[train["Clean_Tweet"] != ""].copy()
validation = validation[validation["Clean_Tweet"] != ""].copy()

print("TRAINING DATA AFTER CLEANING")
print(train[["Tweet", "Clean_Tweet"]].head(10))

print("\nTRAINING SHAPE:", train.shape)
print("VALIDATION SHAPE:", validation.shape)

# Save cleaned datasets
train.to_csv(project_path / "dataset" / "twitter_training_cleaned.csv", index=False)
validation.to_csv(project_path / "dataset" / "twitter_validation_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully!")
