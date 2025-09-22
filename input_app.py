import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io
import plotly.express as px

# Custom CSS for enhanced styling
st.set_page_config(
    page_title="Comment Submission",
    page_icon="💬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Container */
    .input-container {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Sentiment Chips */
    .sentiment-chip {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .positive-chip {
        background-color: #e6f3ea;
        color: #188038;
    }
    
    .negative-chip {
        background-color: #fce8e6;
        color: #d93025;
    }
    
    .neutral-chip {
        background-color: #f1f3f4;
        color: #5f6368;
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
    if res.status_code in [200, 201]:
        st.success("Comment submitted ✅ Sentiment updated in GitHub CSV")
    else:
        st.error(f"Failed to update CSV: {res.text}")

# Main App Layout
st.title("💬 Comment Submission Portal")

# Sidebar for Additional Information
st.sidebar.header("📊 Submission Insights")

# Fetch current data for sidebar stats
df, _ = get_csv()

# Sidebar Metrics
st.sidebar.metric("Total Comments", len(df))
sentiment_counts = df['sentiment'].value_counts()
st.sidebar.metric("Positive Comments", sentiment_counts.get('positive', 0))
st.sidebar.metric("Negative Comments", sentiment_counts.get('negative', 0))

# Sentiment Distribution Chart
st.sidebar.subheader("Sentiment Distribution")
fig = px.pie(
    values=sentiment_counts.values, 
    names=sentiment_counts.index, 
    color_discrete_sequence=['#188038', '#d93025', '#5f6368']
)
st.sidebar.plotly_chart(fig, use_container_width=True)

# Main Input Container
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# User Input Columns
col1, col2 = st.columns([2, 1])

with col1:
    user_id = st.text_input(
        "Enter your User ID", 
        placeholder="username or email",
        help="This helps us track and categorize comments"
    )

with col2:
    # Optional: Add a profile picture upload or avatar selection
    st.write("👤 Profile")

# Comment Input
user_comment = st.text_area(
    "Share your thoughts", 
    placeholder="Type your comment here...",
    height=200
)

# Submit Button
submit_col1, submit_col2 = st.columns([3, 1])

with submit_col2:
    submit_button = st.button("Submit Comment", use_container_width=True)

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

    # Problem Summary
    def summarize_problem(text, sentiment_label, max_words=12):
        if sentiment_label.lower() == "negative":
            words = text.split()
            return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
        return ""

    problem_summary = summarize_problem(user_comment, sentiment)

    # Prepare New Row
    new_row = {
        "user_id": user_id or "Anonymous",
        "comment": user_comment,
        "sentiment": sentiment,
        "score": score,
        "ProblemSummary": problem_summary
    }

    # Update DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Sentiment Visualization
    sentiment_color = (
        "green" if sentiment.lower() == 'positive' 
        else "red" if sentiment.lower() == 'negative' 
        else "gray"
    )

    # Result Display
    st.markdown(f"""
    <div class="sentiment-chip {sentiment_color}-chip">
        Sentiment: {sentiment} (Confidence: {score:.2f})
    </div>
    """, unsafe_allow_html=True)

    if problem_summary:
        st.info(f"Key Problem Summary: {problem_summary}")

    # Update GitHub CSV
    if _:
        update_csv(df, _)
    else:
        st.error("Could not fetch CSV SHA from GitHub.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Powered by AI-Driven Sentiment Analysis</div>", unsafe_allow_html=True)
