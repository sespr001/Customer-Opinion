# ==========================================================
# 📊 Streamlit App: Customer Opinion Sentiment Analysis
# ==========================================================

import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import pipeline

# --- Load NLP models ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")

# --- Streamlit page config ---
st.set_page_config(page_title="Customer Opinion Analyzer", layout="wide")
st.title("🧠 Customer Opinion Sentiment Analysis App")

# ==========================================================
# 1️⃣ File upload
# ==========================================================
st.header("Upload CSV File with 20 Customer Opinions")
uploaded_file = st.file_uploader("Upload your CSV file (must have one column named 'opinion')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ==========================================================
    # 2️⃣ Text preprocessing
    # ==========================================================
    st.header("Text Cleaning and Preprocessing")

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    df["cleaned_text"] = df["opinion"].apply(clean_text)
    st.dataframe(df.head())

    # ==========================================================
    # 3️⃣ Word Cloud & Top 10 Words
    # ==========================================================
    st.header("Word Cloud & Top 10 Most Frequent Words")

    all_words = " ".join(df["cleaned_text"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    st.subheader("Word Cloud")

    # --- Create and show the Word Cloud ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Customer Opinions", fontsize=14, weight='bold')
    st.pyplot(fig)


    # --- Top 10 words ---
    word_counts = Counter(all_words.split())
    top_words = word_counts.most_common(10)
    words, counts = zip(*top_words)

    st.subheader("Top 10 Words Bar Chart")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(words, counts, color='skyblue')
    ax.set_title("Top 10 Most Frequent Words", fontsize=14, weight='bold')
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==========================================================
    # 4️⃣ Additional Analysis: Adjectives
    # ==========================================================
    st.header("Additional Analysis – Most Frequent Adjectives")

    adjectives = []
    for text in df["cleaned_text"]:
        doc = nlp(text)
        adjectives += [token.text for token in doc if token.pos_ == "ADJ"]

    adj_counts = Counter(adjectives).most_common(10)
    if adj_counts:
        adj_words, adj_freqs = zip(*adj_counts)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(adj_words, adj_freqs, color='orange')
        ax.set_title("Most Frequent Adjectives in Customer Opinions", fontsize=14, weight='bold')
        ax.set_xlabel("Adjectives")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No adjectives found in the opinions.")

    # ==========================================================
    # 5️⃣ Sentiment Analysis
    # ==========================================================
    st.header("Sentiment Classification (Positive / Neutral / Negative)")

    sentiment_analyzer = pipeline("sentiment-analysis",
                                  model="nlptown/bert-base-multilingual-uncased-sentiment")

    results = []
    for text in df["opinion"]:
        result = sentiment_analyzer(text)[0]
        results.append(result)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    def simplify_sentiment(label):
        stars = int(label.split()[0])
        if stars <= 2:
            return "negative"
        elif stars == 3:
            return "neutral"
        else:
            return "positive"

    df["sentiment_category"] = df["sentiment_label"].apply(simplify_sentiment)

    st.dataframe(df[["opinion", "sentiment_category", "sentiment_score"]])

    # --- Sentiment distribution plot ---
    sent_counts = df["sentiment_category"].value_counts()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.bar(sent_counts.index, sent_counts.values, color=["red", "gray", "green"])
    ax.set_title("Sentiment Distribution of Customer Opinions", fontsize=14, weight='bold')
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Opinions")
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

    # ==========================================================
    # 6️⃣ User Input for New Opinion
    # ==========================================================
    st.header("Try It Yourself – Enter a New Opinion")

    user_text = st.text_input("Write a new customer opinion:")
    if user_text:
        result = sentiment_analyzer(user_text)[0]
        stars = int(result["label"].split()[0])
        score = result["score"]

        if stars <= 2:
            category = "negative 😡"
        elif stars == 3:
            category = "neutral 😐"
        else:
            category = "positive 😊"

        st.success(f"**Sentiment:** {category} (Confidence: {score:.2f})")

else:
    st.warning("👆 Please upload a CSV file to start the analysis.")
