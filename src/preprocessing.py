import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# Load Dataset

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

print("Datasets Loaded Successfully")
print()


# Add Labels

fake_df["label"] = 0
true_df["label"] = 1


# Merge Dataset

df = pd.concat([fake_df, true_df], ignore_index=True)

print("Dataset Merged Successfully")
print(f"Total Records : {len(df)}")
print()


# Shuffle Dataset

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Shuffled")
print()


# Remove Unnecessary Columns

df = df[["title", "text", "label"]]


# Combine Title + Text

df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]


# Text Cleaning

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Cleaning Text...")
print()

df["content"] = df["content"].apply(clean_text)

print("Cleaning Completed")
print()


# Save Processed Dataset

df.to_csv("dataset/processed_news.csv", index=False)

print("=" * 50)
print("Preprocessing Completed Successfully")
print("=" * 50)

print()

print(df.head())

print()

print("Dataset Shape :", df.shape)

print()

print(df["label"].value_counts())