"""Shared text-cleaning helpers used for training and prediction."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Download the small NLTK datasets required by this project if absent."""
    for resource, package in (("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Normalize an article into the same form used by the TF-IDF model."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(PUNCTUATION_TABLE)

    return " ".join(
        LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS
    )
