import pickle
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load Saved Model & Vectorizer

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Text Cleaning Function

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


# Prediction Function

def predict_news(news):

    cleaned_news = clean_text(news)

    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)[0]

    if prediction == 0:
        return "Fake News"
    else:
        return "Real News"



# Test Prediction

if __name__ == "__main__":

    print("=" * 60)
    print("Fake News Detection System")
    print("=" * 60)

    news = input("\nEnter News:\n\n")

    result = predict_news(news)

    print("\nPrediction :", result)