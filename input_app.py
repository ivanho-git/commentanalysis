import streamlit as st
import pandas as pd
import pickle
import re

# ---------------------------
# Load trained model + vectorizer
# ---------------------------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# Streamlit page
# ---------------------------
st.title("💬 Comment Input Page (Classical ML, No SpaCy)")

# CSV to store comments
csv_file = "comments.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary"])

# ---------------------------
# User input
# ---------------------------
user_comment = st.text_area("Enter your comment:")

if st.button("Submit"):
    if user_comment.strip() != "":

        # ---------------------------
        # Clean text
        # ---------------------------
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#\w+", "", text)
            text = re.sub(r"[^a-z\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        cleaned = clean_text(user_comment)

        # ---------------------------
        # Vectorize + Predict
        # ---------------------------
        vec = vectorizer.transform([cleaned])
        sentiment = model.predict(vec)[0]
        score = max(model.predict_proba(vec)[0])

        # ---------------------------
        # Simple Problem Summary for negative comments
        # ---------------------------
        def summarize_problem(text, sentiment_label, max_words=12):
            if sentiment_label.lower() == "negative":
                words = text.split()
                return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
            return ""

        problem_summary = summarize_problem(user_comment, sentiment)

        # ---------------------------
        # Save to CSV
        # ---------------------------
        new_row = {
            "comment": user_comment,
            "sentiment": sentiment,
            "score": score,
            "ProblemSummary": problem_summary
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)

        st.success(f"Comment submitted ✅ Sentiment: {sentiment} ({score:.2f})")
        if problem_summary:
            st.info(f"Key Problem Summary: {problem_summary}")
