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
        st.success("✅ Comment submitted — Sentiment updated in GitHub CSV")
    else:
        st.error(f"❌ Failed to update CSV: {res.text}")

# ---------------------------
# Inject Dark Theme CSS
# ---------------------------
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .big-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #FFD700;
            text-shadow: 2px 2px 8px #FF4500;
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #BBBBBB;
            margin-bottom: 30px;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea {
            background-color: #1E1E1E !important;
            color: #FAFAFA !important;
            border: 1px solid #444 !important;
            border-radius: 10px;
        }
        .stButton button {
            background: linear-gradient(90deg, #FF0080, #7928CA);
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #7928CA, #FF0080);
            transform: scale(1.05);
        }
        .sentiment-box {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .positive { background-color: rgba(0, 200, 0, 0.2); color: #00FF7F; }
        .negative { background-color: rgba(200, 0, 0, 0.2); color: #FF6347; }
        .neutral { background-color: rgba(200, 200, 0, 0.2); color: #FFD700; }
        footer {
            text-align: center;
            margin-top: 40px;
            color: #AAAAAA;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Page Title
# ---------------------------
st.markdown('<div class="big-title">💡 Sentilytics 💡</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">📊 Made by the people, for the people 📊</div>', unsafe_allow_html=True)

df, sha = get_csv()

# ---------------------------
# User input
# ---------------------------
user_id = st.text_input("👤 Enter your User ID:", value="Unknown")
user_comment = st.text_area("💬 Enter your comment:")

if st.button("🚀 Submit") and user_comment.strip() != "":

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
        "user_id": user_id,
        "comment": user_comment,
        "sentiment": sentiment,
        "score": score,
        "ProblemSummary": problem_summary
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if sha:
        update_csv(df, sha)
    else:
        st.error("⚠️ Could not fetch CSV SHA from GitHub.")

    # ---------------------------
    # Display result with colored box
    # ---------------------------
    sentiment_class = "neutral"
    if sentiment.lower() == "positive":
        sentiment_class = "positive"
    elif sentiment.lower() == "negative":
        sentiment_class = "negative"

    st.markdown(
        f'<div class="sentiment-box {sentiment_class}">Sentiment: {sentiment} ({score:.2f})</div>',
        unsafe_allow_html=True
    )

    if problem_summary:
        st.markdown(f'<div class="sentiment-box negative">⚠️ Key Problem: {problem_summary}</div>', unsafe_allow_html=True)

# ---------------------------
# Footer Branding
# ---------------------------
st.markdown(
    """<footer> 
    @2025💖 Made with love by <b>Team CodeBlooded</b></footer>""",
    unsafe_allow_html=True
)
