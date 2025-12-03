import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from transformers import pipeline

# ------------------ Setup ------------------
analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_bert_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

bert_pipeline = load_bert_pipeline()

st.set_page_config(page_title="VADER + BERT + ENSEMBLE Sentiment Analyzer", layout="wide")
st.title("🧠 VADER + BERT + ENSEMBLE Sentiment Analysis App")

# ------------------ Helpers ------------------
def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_vader_compound(text):
    return analyzer.polarity_scores(str(text))['compound']

def get_bert_result(text):
    text = str(text).strip()
    if not text:
        return {"label": "neutral", "score": 0.0}
    text = text[:2000]
    res = bert_pipeline(text, truncation=True, max_length=512)[0]
    return {"label": res["label"].lower(), "score": float(res["score"])}

# ------------------ Ensemble ------------------
def ensemble_prediction(vader_comp, bert_label):
    # Convert VADER compound to label
    if vader_comp >= 0.05:
        vader_label = 2
    elif vader_comp <= -0.05:
        vader_label = 0
    else:
        vader_label = 1

    # Convert BERT label to numeric
    if bert_label.lower() == "positive":
        bert_numeric = 2
    elif bert_label.lower() == "negative":
        bert_numeric = 0
    else:
        bert_numeric = 1

    # Weighted Ensemble → BERT = 0.7, VADER = 0.3
    final_score = (0.7 * bert_numeric) + (0.3 * vader_label)
    final_label = round(final_score)

    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping[final_label]

def common_words(df, sentiment_col, sentiment_type):
    text_data = df[df[sentiment_col] == sentiment_type]['reviewText'].dropna().astype(str)
    all_words = []
    for text in text_data:
        all_words.extend(clean_text(text))
    word_counts = Counter(all_words)
    return word_counts.most_common(10)

# ------------------ Sidebar ------------------
option = st.sidebar.radio("Choose Input Type", ["📝 Single Text", "📂 CSV File"])

# ------------------ Single Text Mode ------------------
if option == "📝 Single Text":
    user_input = st.text_area("✏️ Enter a review:")
    if st.button("🔍 Analyze"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            vader_score = analyzer.polarity_scores(user_input)
            vader_sentiment = get_vader_sentiment(user_input)
            bert_res = get_bert_result(user_input)

            # Ensemble Result
            ensemble_final = ensemble_prediction(vader_score['compound'], bert_res['label'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🟦 VADER Result")
                st.write("Sentiment:", vader_sentiment.upper())
                st.write("Compound Score:", vader_score['compound'])

            with col2:
                st.subheader("🟩 BERT Result")
                st.write("Sentiment:", bert_res['label'].upper())
                st.write("Confidence:", f"{bert_res['score']:.2f}")

            with col3:
                st.subheader("🟪 Ensemble (VADER + BERT)")
                st.write("Final Sentiment:", ensemble_final.upper())

# ------------------ CSV Mode ------------------
else:
    uploaded_file = st.file_uploader("📂 Upload CSV (must have `reviewText` column)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'reviewText' not in df.columns:
            st.error("❌ 'reviewText' column not found.")
        else:
            analysis_type = st.radio(
                "Choose Sentiment Model",
                ["VADER Only", "BERT Only", "Both VADER & BERT", "ENSEMBLE"]
            )

            if st.button("🚀 Run Analysis"):
                with st.spinner("🔄 Running sentiment analysis..."):

                    # VADER
                    if analysis_type in ["VADER Only", "Both VADER & BERT", "ENSEMBLE"]:
                        df['vader_sentiment'] = df['reviewText'].apply(get_vader_sentiment)
                        df['vader_compound'] = df['reviewText'].apply(get_vader_compound)

                    # BERT
                    if analysis_type in ["BERT Only", "Both VADER & BERT", "ENSEMBLE"]:
                        bert_out = df['reviewText'].apply(get_bert_result)
                        df['bert_label'] = bert_out.apply(lambda x: x['label'])
                        df['bert_score'] = bert_out.apply(lambda x: x['score'])

                    # Ensemble
                    if analysis_type == "ENSEMBLE":
                        df['ensemble_sentiment'] = df.apply(
                            lambda row: ensemble_prediction(row['vader_compound'], row['bert_label']),
                            axis=1
                        )

                st.success("✅ Sentiment analysis complete!")
                st.dataframe(df.head())

                # ---------- Visualizations ----------
                # VADER
                if "vader_sentiment" in df.columns:
                    st.header("📊 VADER Analysis")
                    fig, ax = plt.subplots()
                    sns.countplot(x='vader_sentiment', data=df, ax=ax)
                    st.pyplot(fig)

                # BERT
                if "bert_label" in df.columns:
                    st.header("📊 BERT Analysis")
                    fig, ax = plt.subplots()
                    sns.countplot(x='bert_label', data=df, ax=ax)
                    st.pyplot(fig)

                # Ensemble
                if "ensemble_sentiment" in df.columns:
                    st.header("🧩 Ensemble Model Analysis")
                    fig, ax = plt.subplots()
                    sns.countplot(x='ensemble_sentiment', data=df, ax=ax)
                    st.pyplot(fig)

                # Download
                st.download_button(
                    "📥 Download Results",
                    df.to_csv(index=False).encode('utf-8'),
                    "sentiment_results.csv",
                    "text/csv"
                )
