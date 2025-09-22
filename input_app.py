import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io  # Already added

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
        return df, content["sha"]
    else:
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary"]), None

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
    # Improved Aspect-Based Problem Summary for negative comments
    # ---------------------------
    def summarize_problem(text, sentiment_label, max_aspects=3, max_words_per_aspect=5):
        if sentiment_label.lower() == "negative":
            # Define common negative adjectives
            negative_adjectives = ['bad', 'poor', 'terrible', 'worst', 'awful', 'horrible']
            
            # Use regex to find patterns: <negative_adjective> followed by words (potential nouns/aspects)
            aspects = []
            for adj in negative_adjectives:
                pattern = re.compile(rf"\b{adj}\s+([\w\s]+?)(?=\s+\w+|$)", re.IGNORECASE)  # Matches adjective + following words
                matches = pattern.findall(text.lower())
                for match in matches:
                    # Clean and limit each aspect to max_words_per_aspect
                    aspect_words = match.split()[:max_words_per_aspect]
                    if aspect_words:  # Only add if there's content
                        aspects.append(" ".join(aspect_words))
            
            if aspects:
                # Create a summary from the extracted aspects
                unique_aspects = list(set(aspects))  # Remove duplicates
                summary = "Key problems: " + "; ".join(unique_aspects[:max_aspects]) + "..."  # Limit to max_aspects
                return summary
            else:
                # Fallback: Shorten the original text if no aspects found
                words = text.split()
                return " ".join(words[:10]) + "..."  # First 10 words as fallback
        return ""

    problem_summary = summarize_problem(user_comment, sentiment)

    # ---------------------------
    # Save to GitHub CSV
    # ---------------------------
    new_row = {
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
