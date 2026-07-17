import streamlit as st

from src.predict import predict_news


st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.caption("Paste a news headline or article to classify it with the trained model.")

news = st.text_area("News text", height=220, placeholder="Paste news content here...")
if st.button("Analyze", type="primary"):
    try:
        result = predict_news(news)
        st.success(f"Prediction: {result}")
        st.caption("This is a model prediction, not a fact-check.")
    except ValueError as error:
        st.warning(str(error))
