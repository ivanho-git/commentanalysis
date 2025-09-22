
import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io

# Page Configuration
st.set_page_config(
    page_title="Comment Submission",
    page_icon="💬",
    layout="wide"
)

# Advanced Custom CSS
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Roboto', 'Inter', sans-serif;
    }

    /* Main Container */
    .main-container {
        background: white;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        padding: 30px;
        max-width: 800px;
        margin: 0 auto;
        transition: all 0.3s ease;
    }

    .main-container:hover {
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        transform: translateY(-5px);
    }

    /* Input Styling */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 1.5px solid #e0e0e0;
        padding: 12px;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }

    /* Submit Button */
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        padding: 12px 24px;
    }

    .stButton > button:hover {
        background-color: #357abd;
        transform: scale(1.05);
    }

    /* Success Message */
    .success-message {
        background-color: #e6f3ea;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model + vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# GitHub API setup
TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = st.secrets["GITHUB_REPO"]
CSV_PATH = st.secrets["CSV_PATH"]

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
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))
        if "user_id" not in df.columns:
            df["user_id"] = "Unknown"
        return df, content["sha"]
    else:
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary", "user_id"]), None

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
    return res.status_code in [200, 201]

# Main App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title with Modern Typography
st.markdown("""
    <h1 style="
        text-align: center; 
        color: #333; 
        font-weight: 700; 
        margin-bottom: 30px;
        background: linear-gradient(to right, #4a90e2, #50c878);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">
    💬 Share Your Thoughts
    </h1>
""", unsafe_allow_html=True)

# User Input Columns
col1, col2 = st.columns([3, 1])

with col1:
    user_id = st.text_input(
        "User ID", 
        placeholder="Enter your username",
        help="This helps us track your comments"
    )

with col2:
    # Placeholder for future profile features
    st.write("👤 Profile")

# Comment Input
user_comment = st.text_area(
    "Your Comment", 
    placeholder="What's on your mind?",
    height=200
)

# Submit Button
submit_col1, submit_col2 = st.columns([3, 1])

with submit_col2:
    submit_button = st.button("Submit", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Submission Logic
if submit_button and user_comment.strip() != "":
    # Text Cleaning Function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Sentiment Analysis
    cleaned = clean_text(user_comment)
    vec = vectorizer.transform([cleaned])
    sentiment = model.predict(vec)[0]
    score = max(model.predict_proba(vec)[0])

    # Prepare New Row
    new_row = {
        "user_id": user_id or "Anonymous",
        "comment": user_comment,
        "sentiment": sentiment,
        "score": score,
        "ProblemSummary": ""
    }

    # Fetch current DataFrame
    df, sha = get_csv()

    # Update DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Update GitHub CSV
    if sha:
        success = update_csv(df, sha)
        
        # Success Message with Modern Styling
        if success:
            st.markdown("""
            <div class="success-message">
                <h3>✅ Comment Uploaded Successfully!</h3>
                <p>Thank you for sharing your thoughts.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to upload comment. Please try again.")
    else:
        st.error("Could not fetch CSV SHA from GitHub.")

# Footer with Modern Design
st.markdown("""
<div style="
    text-align: center; 
    color: #666; 
    margin-top: 30px;
    padding: 15px;
    background-color: rgba(255,255,255,0.7);
    border-radius: 10px;
">
    Powered by AI-Driven Insights | © 2024 Feedback Platform
</div>

