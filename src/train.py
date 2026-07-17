import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load Dataset

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/processed_news.csv")

# TF-IDF accepts text only.  Remove missing or blank articles in case the
# processed dataset was created by an earlier preprocessing run.
df = df.dropna(subset=["content", "label"])
df = df[df["content"].astype(str).str.strip().ne("")].copy()
df["content"] = df["content"].astype(str)

print("Dataset Loaded Successfully")
print(df.shape)
print()


# Features & Labels

X = df["content"]

y = df["label"]


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print()


# TF-IDF

print("=" * 60)
print("Applying TF-IDF...")
print("=" * 60)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print("TF-IDF Completed")
print()


# Models

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Multinomial Naive Bayes":
        MultinomialNB(),

    "Passive Aggressive":
        PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=42
        )

}

best_model = None

best_accuracy = 0

best_name = ""


# Training

print("=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining : {name}")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name

print("=" * 60)
print("Training Completed")
print("=" * 60)


# Best Model

print()

print("Best Model :", best_name)

print("Accuracy   :", round(best_accuracy * 100, 2), "%")

print()

# Classification Report

prediction = best_model.predict(X_test)

print(classification_report(y_test, prediction))


# Save Model
pickle.dump(
    best_model,
    open("models/model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("models/vectorizer.pkl", "wb")
)

print("=" * 60)
print("Model Saved Successfully")
print("=" * 60)
