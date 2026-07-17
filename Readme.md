# Fake News Detection

Classifies English news text as fake or real using TF-IDF features and a supervised machine-learning model.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place `Fake.csv` and `True.csv` in `dataset/`. Both files must contain `title` and `text` columns.

## Run

```powershell
python src/preprocessing.py
python src/train.py
python src/predict.py
streamlit run app.py
```

The first run downloads the NLTK `stopwords` and `wordnet` resources. The application prediction is a statistical estimate and should not be treated as a factual verification.
