import streamlit as st
import pandas as pd
import requests
import base64
import json
import io

# Define colors for a YouTube-like UI
bg_color = "#141414"
text_color = "#FFFFFF"
comment_bg = "#1E1E1E"
author_color = "#AAAAAA"
timestamp_color = "#777777"
like_color = "#F00"  # Red for likes

# Apply global styling
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
h1, h2 {{
    color: {text_color};
}}
.stExpander {{
    border: none;
    background-color: {comment_bg};
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
}}
.comment-container {{
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}}
.author-info {{
    margin-right: 10px;
    font-size: 0.8em;
    color: {author_color};
}}
.author-name {{
    font-weight: bold;
}}
.timestamp {{
    color: {timestamp_color};
}}
.comment-text {{
    flex-grow: 1;
}}
.like-button {{
    background-color: transparent;
    border: none;
    color: {like_color};
    font-size: 1em;
    cursor: pointer;
    padding: 0;
}}
</style>
""", unsafe_allow_html=True)

st.title("📝 Comment Output Page ")

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
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))
        return df
    else:
        st.error(f"Failed to fetch CSV from GitHub: {res.status_code}")
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary", "user_id"])

# Add a button to refresh data
if st.button("Refresh Data"):
    st.session_state.df = get_csv()
else:
    if 'df' not in st.session_state:
        st.session_state.df = get_csv()

df = st.session_state.df

# Display summary at the top
st.header("Comment Overview")
if not df.empty:
    st.write(f"**Total Comments:** {len(df)}")
else:
    st.info("No comments available yet.")

# Display comments in a YouTube-like layout
if not df.empty:
    for idx, row in df[::-1].iterrows():
        with st.expander(f"Comment {len(df) - idx}", expanded=False):
            col1, col2 = st.columns([2, 8])  # Adjust column widths

            with col1:
                if "user_id" in df.columns and not pd.isna(row["user_id"]) and row["user_id"]:
                    st.markdown(f"<div class='author-info'><span class='author-name'>{row['user_id']}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='author-info'>Unknown</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"<div class='comment-text'>{row['comment']}</div>", unsafe_allow_html=True)
                if not pd.isna(row['sentiment']) and row['sentiment']:
                    sentiment_color = "green" if row['sentiment'].lower() == "positive" else "red"
                    st.markdown(f"<span style='color:{sentiment_color}'>Sentiment: {row['sentiment']} | Score: {row['score']:.2f}</span>", unsafe_allow_html=True)
                if "ProblemSummary" in df.columns and not pd.isna(row["ProblemSummary"]) and row["ProblemSummary"]:
                    st.markdown(f"**Problem Summary:** {row['ProblemSummary']}")

                # Add a like button (placeholder)
                st.markdown("<button class='like-button'>👍</button>", unsafe_allow_html=True)

            st.markdown("---")
else:
    st.info("No comments yet.")
