
# 📦 IMPORTS

# Standard library
import re

# Third-party libraries
import numpy as np
import pandas as pd
import joblib
import nltk
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup

# NLTK tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==============================
# ⚙️ CONFIGURATION
# ==============================

class Config:
    MODEL_PATH = 'model/passmodel.pkl'
    TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
    DATA_PATH = 'data/custom_dataset.csv'


# ==============================
# 📥 DOWNLOAD NLTK RESOURCES
# ==============================

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


# ==============================
# 🤖 LOAD MODEL
# ==============================

@st.cache_resource
def load_model(model_path, tokenizer_path):
    try:
        vectorizer = joblib.load(tokenizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


# ==============================
# 🧹 TEXT CLEANING
# ==============================

def clean_text(text, stop_words, lemmatizer):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# ==============================
# 💊 DRUG RECOMMENDATION
# ==============================

def top_drugs_extractor(condition, df):
    filtered_df = df[
        (df['rating'] >= 9) & (df['usefulCount'] >= 90)
    ].sort_values(by=['rating', 'usefulCount'], ascending=False)

    drugs = (
        filtered_df[filtered_df['condition'] == condition]['drugName']
        .drop_duplicates()
        .head(4)
        .tolist()
    )

    return drugs


# ==============================
# 🔍 PREDICTION FUNCTION
# ==============================

def predict(text, vectorizer, model, data_path, stop_words, lemmatizer):
    cleaned = clean_text(text, stop_words, lemmatizer)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    df = pd.read_csv(data_path)
    drugs = top_drugs_extractor(prediction, df)

    return prediction, drugs


# ==============================
# 🎨 MAIN APP
# ==============================

def main():
    st.set_page_config(
        page_title='DPDR',
        page_icon='👨‍⚕️',
        layout='wide'
    )

    # Download NLTK data
    download_nltk_resources()

    # Initialize NLP tools
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # Load model
    vectorizer, model = load_model(
        Config.MODEL_PATH,
        Config.TOKENIZER_PATH
    )

    if vectorizer is None or model is None:
        return

    # ================= UI =================

    st.title("💉 Disease Prediction & Drug Recommendation System")
    st.markdown("---")

    # Input
    raw_text = st.text_area(
        "📝 Enter Symptoms",
        placeholder="e.g. fever, headache, nausea..."
    )

    # Predict button
    if st.button("🔍 Predict"):

        if not raw_text.strip():
            st.warning("⚠️ Please enter symptoms")
            return

        with st.spinner("🧠 Analyzing..."):
            condition, drugs = predict(
                raw_text,
                vectorizer,
                model,
                Config.DATA_PATH,
                stop_words,
                lemmatizer
            )

        # ================= RESULTS =================

        st.markdown("---")
        st.subheader("🎯 Predicted Condition")
        st.success(condition)

        st.markdown("---")
        st.subheader("💊 Recommended Drugs")

        if not drugs:
            st.warning("No drugs found for this condition")
        else:
            colors = ["#b4befe", "#a6e3a1", "#f38ba8", "#f2cdcd"]

            for i, drug in enumerate(drugs):
                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        border-radius:8px;
                        background-color:{colors[i % len(colors)]};
                        margin:5px;">
                        <b>{i+1}. {drug}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ================= CHART =================

        if drugs:
            st.markdown("---")
            st.subheader("📊 Drug Visualization")

            df_chart = pd.DataFrame({
                "Drug": drugs,
                "Rank": list(range(len(drugs), 0, -1))
            })

            fig = px.bar(
                df_chart,
                x="Rank",
                y="Drug",
                orientation='h',
                title="Top Recommended Drugs"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ================= DISCLAIMER =================

    st.markdown("---")
    st.warning("""
    ⚠️ This is a prototype. Always consult a doctor for medical advice.
    """)


# ==============================
# 🚀 RUN APP
# ==============================

if __name__ == "__main__":
    main()
