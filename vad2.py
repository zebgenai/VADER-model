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

st.set_page_config(page_title="VADER + BERT Sentiment Analyzer", layout="wide")
st.title("🧠 VADER + BERT Sentiment Analysis App")

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
        return {"label": "NEUTRAL", "score": 0.0}
    text = text[:2000]
    res = bert_pipeline(text, truncation=True, max_length=512)[0]
    return {"label": res["label"], "score": float(res["score"])}

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
            b = get_bert_result(user_input)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("VADER")
                st.write("Sentiment:", vader_sentiment.upper())
                st.write("Compound Score:", vader_score['compound'])
                st.json(vader_score)
            with col2:
                st.subheader("BERT")
                st.write("Sentiment:", b['label'])
                st.write("Confidence:", f"{b['score']:.2f}")

# ------------------ CSV Mode ------------------
else:
    uploaded_file = st.file_uploader("📂 Upload CSV (must have `reviewText` column)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'reviewText' not in df.columns:
            st.error("❌ 'reviewText' column not found.")
        else:
            analysis_type = st.radio("Choose Sentiment Model", ["VADER Only", "BERT Only", "Both VADER & BERT"])

            if st.button("🚀 Run Analysis"):
                with st.spinner("🔄 Running sentiment analysis..."):

                    if analysis_type in ["VADER Only", "Both VADER & BERT"]:
                        df['vader_sentiment'] = df['reviewText'].apply(get_vader_sentiment)
                        df['vader_compound'] = df['reviewText'].apply(get_vader_compound)

                    if analysis_type in ["BERT Only", "Both VADER & BERT"]:
                        bert_out = df['reviewText'].apply(get_bert_result)
                        df['bert_label'] = bert_out.apply(lambda x: x['label'])
                        df['bert_score'] = bert_out.apply(lambda x: x['score'])

                st.success("✅ Sentiment analysis complete!")
                st.dataframe(df.head())

                # ===== VADER Section =====
                if "vader_sentiment" in df.columns:
                    st.header("📊 VADER Analysis")
                    fig, ax = plt.subplots()
                    sns.countplot(x='vader_sentiment', data=df, ax=ax)
                    st.pyplot(fig)

                    fig, ax = plt.subplots()
                    counts = df['vader_sentiment'].value_counts()
                    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
                    st.pyplot(fig)

                    st.subheader("📝 Common Words per Sentiment (VADER)")
                    for sent in ['positive', 'negative', 'neutral']:
                        words = common_words(df, 'vader_sentiment', sent)
                        st.write(f"**{sent.upper()}**: {', '.join([w for w, _ in words])}")

                # ===== BERT Section =====
                if "bert_label" in df.columns:
                    st.header("📊 BERT Analysis")
                    fig, ax = plt.subplots()
                    sns.countplot(x='bert_label', data=df, ax=ax)
                    st.pyplot(fig)

                    fig, ax = plt.subplots()
                    counts = df['bert_label'].value_counts()
                    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
                    st.pyplot(fig)

                    st.subheader("📝 Common Words per Sentiment (BERT)")
                    for sent in df['bert_label'].unique():
                        words = common_words(df, 'bert_label', sent)
                        st.write(f"**{sent.upper()}**: {', '.join([w for w, _ in words])}")

                # Download button
                st.download_button(
                    "📥 Download Results",
                    df.to_csv(index=False).encode('utf-8'),
                    "sentiment_results.csv",
                    "text/csv"
                )
