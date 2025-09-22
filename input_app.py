import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io  # Add this import

# ---------------------------
# Load trained model + vectorizer
# ---------------------------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# GitHub API setup
# ---------------------------
TOKEN = st.secrets["GITHUB_TOKEN"]       # Add your token in Streamlit Secrets
REPO = st.secrets["GITHUB_REPO"]         # e.g., "username/repo"
CSV_PATH = st.secrets["CSV_PATH"]        # e.g., "comments.csv"

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

URL = f"https://api.github.com/repos/{REPO}/contents/{CSV_PATH}"

def get_csv():
    """Fetch CSV content from GitHub and return DataFrame + SHA"""
    res = requests.get(URL, headers=HEADERS)
    if res.status_code == 200:
        content = res.json()
        csv_bytes = base64.b64decode(content["content"])
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))  # As previously fixed
        if "user_id" not in df.columns:  # Add user_id column if it doesn't exist
            df["user_id"] = "Unknown"  # Default value for existing rows
        return df, content["sha"]
    else:
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary", "user_id"]), None  # Include user_id in empty DataFrame

def update_csv(df, sha):
    """Push updated CSV back to GitHub"""
    csv_str = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_str.encode()).decode()
    data = {
        "message": f"Update comments {datetime.utcnow()}",
        "content": content_b64,
        "sha": sha
    }
    res = requests.put(URL, headers=HEADERS, data=json.dumps(data))
    if res.status_code in [200, 201]:
        st.success("Comment submitted ✅ Sentiment updated in GitHub CSV")
    else:
        st.error(f"Failed to update CSV: {res.text}")

# ---------------------------
# Streamlit page
# ---------------------------
st.title("💬 Comment Input Page (GitHub CSV)")

df, sha = get_csv()

# ---------------------------
# User input
# ---------------------------
user_id = st.text_input("Enter your User ID (e.g., username):", value="Unknown")  # Default to "Unknown"

user_comment = st.text_area("Enter your comment:")

if st.button("Submit") and user_comment.strip() != "":

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
    # Save to GitHub CSV
    # ---------------------------
    new_row = {
        "user_id": user_id,  # Add user_id to the new row
        "comment": user_comment,
        "sentiment": sentiment,
        "score": score,
        "ProblemSummary": problem_summary
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if sha:
        update_csv(df, sha)
    else:
        st.error("Could not fetch CSV SHA from GitHub.")

    # Display result
    st.info(f"Sentiment: {sentiment} ({score:.2f})")
    if problem_summary:
        st.info(f"Key Problem Summary: {problem_summary}")
