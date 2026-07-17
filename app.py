import pickle
import re
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Download NLTK Data
# ----------------------------

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ----------------------------
# Text Cleaning
# ----------------------------

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

# ----------------------------
# Prediction
# ----------------------------

def predict_news(news):

    cleaned_news = clean_text(news)

    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)[0]

    return prediction

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write(
    "Paste a complete news article below and click **Predict**."
)

news = st.text_area(
    "Enter News",
    height=250,
    placeholder="Paste the complete news article here..."
)

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news.")
    else:

        result = predict_news(news)

        st.divider()

        if result == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.write("---")

        st.write("### Information")

        st.write(f"**Words Entered :** {len(news.split())}")

        st.write("**Model Used :** Passive Aggressive Classifier")