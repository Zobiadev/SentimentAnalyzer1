import re
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_model.joblib"

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


model = load_model()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type or paste a movie review. The model predicts whether it is "
         "**positive** or **negative**.")

examples = {
    "— choose an example —": "",
    "Positive": "An absolutely brilliant film. The acting was superb and the story kept me hooked till the end.",
    "Negative": "A boring, predictable mess. The plot made no sense and I wanted my money back.",
    "Mixed": "The visuals were stunning but the story was weak and the ending disappointed me.",
}
choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Your review", value=examples[choice], height=160,
                    placeholder="This movie was amazing because...")

if st.button("Analyze sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter a review first.")
    elif len(clean(text).split()) < 3:
        st.warning("Please write at least a few words.")
    else:
        proba = model.predict_proba([clean(text)])[0]  # [negative, positive]
        pos, neg = float(proba[1]), float(proba[0])
        if pos >= 0.5:
            st.success(f"😊 Positive  ({pos:.1%} confidence)")
        else:
            st.error(f"😞 Negative  ({neg:.1%} confidence)")
        st.progress(pos, text=f"Positive score: {pos:.1%}")
        c1, c2 = st.columns(2)
        c1.metric("Positive", f"{pos:.1%}")
        c2.metric("Negative", f"{neg:.1%}")

with st.expander("About this project"):
    st.write("Model: TF-IDF (unigrams + bigrams) with Logistic Regression, "
             "trained on 50,000 IMDB movie reviews (Maas et al., 2011). "
             "Built with Python, scikit-learn and Streamlit.")
