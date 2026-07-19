"""Streamlit interface for the fake-news classifier."""

import streamlit as st

from src.predict import model, predict_news


st.set_page_config(page_title="NewsCheck", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 15% 5%, #dbeafe 0, transparent 27%),
                        radial-gradient(circle at 85% 15%, #fef3c7 0, transparent 23%), #f8fafc;
            color: #0f172a;
        }
        .hero {
            padding: 2.6rem 0 1.5rem;
            text-align: center;
        }
        .hero h1 { color: #0f172a; font-size: 3rem; margin-bottom: .3rem; }
        .hero p { color: #475569; font-size: 1.1rem; margin: 0 auto; max-width: 650px; }
        .result-card {
            border-radius: 16px; padding: 1.25rem 1.5rem; margin: .75rem 0;
            border: 1px solid; font-size: 1.3rem; font-weight: 700;
        }
        .fake { background: #fff1f2; border-color: #fda4af; color: #9f1239; }
        .real { background: #ecfdf5; border-color: #6ee7b7; color: #065f46; }
        .summary-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            color: #0f172a;
            margin-top: 0.5rem;
        }
        .summary-card h3 {
            margin: 0 0 0.75rem;
            font-size: 1.05rem;
            letter-spacing: 0.01em;
        }
        .summary-card .summary-value {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }
        .summary-chip {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            color: #1d4ed8;
            font-weight: 700;
            padding: 0.85rem 1rem;
            border-radius: 999px;
            text-align: center;
            min-width: 120px;
        }
        .summary-card p {
            margin: 1rem 0 0;
            color: #475569;
            font-size: 0.95rem;
        }
        .stTextArea textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }
        .stSidebar .css-1d391kg {
            background: rgba(255, 255, 255, 0.88);
            box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.9);
        }
        .stButton > button { border-radius: 10px; font-weight: 650; }
        .css-1q8dd3e.edgvbvh3 {
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Short news example": (
        "The city council approved a new public transport budget after a meeting on Tuesday. "
        "The plan will fund additional buses and improve several neighbourhood routes."
    ),
    "Health claim example": (
        "A social media post claims a household ingredient can instantly cure every illness. "
        "The post provides no sources, study, or advice from qualified health professionals."
    ),
}

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

with st.sidebar:
    st.header("Try an example")
    st.caption("Examples help you see the expected input format.")
    for label, example in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.news_text = example
    if st.button("Clear text", use_container_width=True):
        st.session_state.news_text = ""

    st.divider()
    st.subheader("How to use it")
    st.markdown("1. Paste a headline or article.\n2. Select **Analyze article**.\n3. Treat the result as a screening signal, then fact-check it.")

st.markdown(
    """
    <div class="hero">
        <h1>📰 NewsCheck</h1>
        <p>Quickly screen news text with a machine-learning classifier trained on labelled news articles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([3, 1], gap="large")
with left:
    news = st.text_area(
        "Article or headline",
        key="news_text",
        height=260,
        placeholder="Paste a headline, claim, or full news article here...",
        help="More context usually gives the model a more useful signal.",
    )
with right:
    st.markdown(
        """
        <div class="summary-card">
            <h3>Input summary</h3>
            <div class="summary-value">
                <div class="summary-chip">Words<br><strong>{words}</strong></div>
                <div class="summary-chip">Characters<br><strong>{chars}</strong></div>
            </div>
            <p>Classifier: {classifier}</p>
        </div>
        """.format(words=len(news.split()), chars=len(news), classifier=type(model).__name__),
        unsafe_allow_html=True,
    )

analyze = st.button("Analyze article", type="primary", use_container_width=True)

if analyze:
    try:
        result = predict_news(news)
        is_real = result == "Real News"
        result_class = "real" if is_real else "fake"
        icon = "✓" if is_real else "!"
        st.markdown(
            f'<div class="result-card {result_class}">{icon} Model prediction: {result}</div>',
            unsafe_allow_html=True,
        )
        st.info("Important: this is not a fact-check. Verify claims with reliable sources before sharing or acting on them.")
    except ValueError as error:
        st.warning(str(error))
    except FileNotFoundError as error:
        st.error(str(error))

with st.expander("What this tool can and cannot do"):
    st.markdown(
        """
        It compares your text with patterns learned from labelled examples using TF-IDF text features.
        It cannot verify sources, understand new events perfectly, or determine truth with certainty.
        Use it as one signal alongside checking the publisher, publication date, evidence, and independent reporting.
        """
    )
