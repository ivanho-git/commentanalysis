import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io

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
TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = st.secrets["GITHUB_REPO"]
CSV_PATH = st.secrets["CSV_PATH"]

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

URL = f"https://api.github.com/repos/{REPO}/contents/{CSV_PATH}"

def get_csv():
    res = requests.get(URL, headers=HEADERS)
    if res.status_code == 200:
        content = res.json()
        csv_bytes = base64.b64decode(content["content"])
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))
        if "user_id" not in df.columns:
            df["user_id"] = "Unknown"
        return df, content["sha"]
    else:
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary", "user_id"]), None

def update_csv(df, sha):
    csv_str = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_str.encode()).decode()
    data = {
        "message": f"Update comments {datetime.utcnow()}",
        "content": content_b64,
        "sha": sha
    }
    res = requests.put(URL, headers=HEADERS, data=json.dumps(data))
    if res.status_code in [200, 201]:
        st.success("✅ Comment submitted & analyzed by Sentilytics 🚀")
        st.balloons()
    else:
        st.error(f"🚨 Failed to update CSV: {res.text}")

# ---------------------------
# Custom CSS for Sentilytics Branding
# ---------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #1a2a6c, #b21f1f, #fdbb2d);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        color: white;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .company-title {
        font-size: 42px;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 2px 2px 8px #000000;
    }
    .company-subtitle {
        font-size: 20px;
        font-weight: 500;
        color: #ffffff;
        text-align: center;
        margin-bottom: 5px;
    }
    .company-tagline {
        font-size: 18px;
        font-style: italic;
        color: #00FA9A;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextInput>div>div>input {
        border: 3px solid #FFD700;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextArea textarea {
        border: 3px solid #00FA9A;
        border-radius: 10px;
        background-color: #fffdf5;
        font-size: 16px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background: linear-gradient(90deg, #36d1dc, #5b86e5);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        color: black;
    }
    .positive {background: linear-gradient(45deg,#a8ff78,#78ffd6);}
    .negative {background: linear-gradient(45deg,#ff6a88,#ff99ac);}
    .neutral {background: linear-gradient(45deg,#89f7fe,#66a6ff);}
    .footer {
        font-size: 14px;
        text-align: center;
        margin-top: 50px;
        padding: 15px;
        color: #f0f0f0;
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Branding Header
# ---------------------------
st.markdown('<div class="company-title">💡 Sentilytics 💡</div>', unsafe_allow_html=True)
st.markdown('<div class="company-subtitle">⚡ Turning Comments into Smart Insights ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="company-tagline">"Made by the People, for the People" 🌍</div>', unsafe_allow_html=True)

df, sha = get_csv()

# ---------------------------
# User input
# ---------------------------
user_id = st.text_input("👤 Enter your User ID:", value="Unknown")
user_comment = st.text_area("💬 Enter your Comment:")

if st.button("🚀 Submit Comment"):
    if user_comment.strip() != "":
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#\w+", "", text)
            text = re.sub(r"[^a-z\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        cleaned = clean_text(user_comment)
        vec = vectorizer.transform([cleaned])
        sentiment = model.predict(vec)[0]
        score = max(model.predict_proba(vec)[0])

        def summarize_problem(text, sentiment_label, max_words=12):
            if sentiment_label.lower() == "negative":
                words = text.split()
                return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
            return ""

        problem_summary = summarize_problem(user_comment, sentiment)

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
            st.error("❌ Could not fetch CSV SHA from GitHub.")

        sentiment_class = "positive" if sentiment.lower() == "positive" else ("negative" if sentiment.lower() == "negative" else "neutral")
        st.markdown(
            f'<div class="result-box {sentiment_class}">📊 Sentilytics Analysis → <b>{sentiment}</b> (Score: {score:.2f})</div>',
            unsafe_allow_html=True
        )
        if problem_summary:
            st.markdown(
                f'<div class="result-box negative">⚠️ Problem Summary: {problem_summary}</div>',
                unsafe_allow_html=True
            )

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div class="footer">
        © 2025 Sentilytics · "Made by the People, for the People" 🌍 <br>
        Made with ❤️ by <b>Team CodeBlooded</b>
    </div>
    """,
    unsafe_allow_html=True
)
