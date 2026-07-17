"""Load the trained artifacts and classify news text."""

import pickle
from pathlib import Path

try:  # Supports both `python src/predict.py` and package imports.
    from .text_processing import clean_text
except ImportError:
    from text_processing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "vectorizer.pkl"


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run `python src/train.py` first.")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


model, vectorizer = load_artifacts()


def predict_news(news: str) -> str:
    """Return either ``Fake News`` or ``Real News`` for supplied article text."""
    if not isinstance(news, str) or not news.strip():
        raise ValueError("Please enter some news text to classify.")

    cleaned_news = clean_text(news)
    if not cleaned_news:
        raise ValueError("The text has no usable words after cleaning.")

    prediction = model.predict(vectorizer.transform([cleaned_news]))[0]
    return "Real News" if prediction == 1 else "Fake News"


if __name__ == "__main__":
    print("Fake News Detection System")
    print("Enter an article, then press Enter twice to predict it.")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    print(f"Prediction: {predict_news(' '.join(lines))}")
