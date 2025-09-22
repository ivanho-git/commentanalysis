import streamlit as st
import pandas as pd
import requests
import base64
import json
import io  # Add this import

st.title("📝 Comment Output Page (GitHub CSV)")

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
    """Fetch CSV content from GitHub and return DataFrame"""
    res = requests.get(URL, headers=HEADERS)
    if res.status_code == 200:
        content = res.json()
        csv_bytes = base64.b64decode(content["content"])
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))  # Change this line
        return df
    else:
        st.error(f"Failed to fetch CSV from GitHub: {res.status_code}")
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary"])

# Load comments from GitHub
df = get_csv()

# Display comments in card-like layout
if not df.empty:
    # Show latest comments first
    for idx, row in df[::-1].iterrows():
        with st.container():
            col1, col2 = st.columns([1, 5])
            col1.markdown("👤")  # Placeholder for user
            col2.markdown(f"**Comment:** {row['comment']}")
            col2.markdown(f"**Sentiment:** {row['sentiment']}  |  **Score:** {row['score']:.2f}")
            if "ProblemSummary" in df.columns and row["ProblemSummary"]:
                col2.markdown(f"**Problem Summary:** {row['ProblemSummary']}")
            st.markdown("---")
else:
    st.info("No comments yet.")
