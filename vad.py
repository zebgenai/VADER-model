import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

# Setup
analyzer = SentimentIntensityAnalyzer()
st.set_page_config(page_title="VADER Sentiment Analyzer", layout="wide")
st.title("🧠 VADER Sentiment Analysis App")

# Helper function
def get_sentiment(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Text cleaning for word counts
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

# Sidebar input type
option = st.sidebar.radio("Choose Input Type", ["📝 Single Text", "📂 CSV File"])

if option == "📝 Single Text":
    user_input = st.text_area("✏️ Enter a review:")
    if st.button("🔍 Analyze"):
        score = analyzer.polarity_scores(user_input)
        sentiment = get_sentiment(user_input)
        st.markdown(f"**Sentiment:** `{sentiment.upper()}`")
        st.write("Compound Score:", score['compound'])
        st.json(score)

else:
    uploaded_file = st.file_uploader("📂 Upload CSV (must have `reviewText` column)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'reviewText' not in df.columns:
            st.error("❌ 'reviewText' column not found.")
        else:
            df['sentiment'] = df['reviewText'].apply(get_sentiment)
            df['compound'] = df['reviewText'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

            st.success("✅ Sentiment analysis complete!")
            st.dataframe(df[['reviewText', 'sentiment', 'compound']].head())

            # 📈 Compound Score Summary
            avg_score = df['compound'].mean()
            st.metric("📊 Average Compound Score", f"{avg_score:.3f}")

            # 📊 Count Plot
            st.subheader("🔢 Sentiment Count")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='sentiment', data=df, palette='Set2', ax=ax1)
            st.pyplot(fig1)

            # 📈 Pie Chart
            st.subheader("📎 Sentiment Distribution (%)")
            sentiment_counts = df['sentiment'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

            # 🔤 Top Words in Each Sentiment
            st.subheader("🔡 Common Words in Each Sentiment")
            col1, col2, col3 = st.columns(3)
            for label, col in zip(['positive', 'neutral', 'negative'], [col1, col2, col3]):
                words = []
                reviews = df[df['sentiment'] == label]['reviewText']
                for review in reviews:
                    words.extend(clean_text(str(review)))
                common = Counter(words).most_common(10)
                col.markdown(f"**{label.upper()}**")
                for word, freq in common:
                    col.write(f"{word}: {freq}")

            # ⬇️ Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Results as CSV", csv, file_name="vader_sentiment_results.csv", mime="text/csv")
