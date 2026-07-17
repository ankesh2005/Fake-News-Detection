import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "processed_news.csv"
MODELS_DIR = ROOT_DIR / "models"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

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
    max_features=10_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
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

    "Linear SVM":
        SGDClassifier(
            loss="hinge",
            penalty=None,
            max_iter=1000,
            random_state=42,
            tol=1e-3,
        )

}

best_model = None

best_f1 = -1

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

    precision = precision_score(y_test, prediction, zero_division=0)

    recall = recall_score(y_test, prediction, zero_division=0)

    f1 = f1_score(y_test, prediction, zero_division=0)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()

    # F1 balances precision and recall, which is more useful than accuracy
    # alone if a future dataset becomes class-imbalanced.
    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_name = name

print("=" * 60)
print("Training Completed")
print("=" * 60)


# Best Model

print()

print("Best Model :", best_name)

print("F1 Score   :", round(best_f1 * 100, 2), "%")

print()

# Classification Report

prediction = best_model.predict(X_test)

print(classification_report(y_test, prediction, zero_division=0))


# Save Model
MODELS_DIR.mkdir(exist_ok=True)
with (MODELS_DIR / "model.pkl").open("wb") as file:
    pickle.dump(best_model, file)

with (MODELS_DIR / "vectorizer.pkl").open("wb") as file:
    pickle.dump(vectorizer, file)

print("=" * 60)
print("Model Saved Successfully")
print("=" * 60)
