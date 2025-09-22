import streamlit as st
import pandas as pd
import requests
import base64
import json
import io  # Already added

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
        df = pd.read_csv(io.StringIO(csv_bytes.decode()))  # As previously fixed
        return df
    else:
        st.error(f"Failed to fetch CSV from GitHub: {res.status_code}")
        return pd.DataFrame(columns=["comment", "sentiment", "score", "ProblemSummary", "user_id"])

# Add a button to refresh data
if st.button("Refresh Data"):
    st.session_state.df = get_csv()  # Store in session state for refresh
else:
    if 'df' not in st.session_state:
        st.session_state.df = get_csv()

df = st.session_state.df  # Use session state for persistence

# Display summary at the top
st.header("Comment Overview")
if not df.empty:
    st.write(f"**Total Comments:** {len(df)}")
else:
    st.info("No comments available yet.")

# Display comments in an improved, collapsible card-like layout
if not df.empty:
    # Show latest comments first
    for idx, row in df[::-1].iterrows():
        with st.expander(f"Comment {len(df) - idx}: {row['sentiment']} Sentiment"):  # Collapsible expander
            col1, col2 = st.columns([1, 5])
            
            # Enhanced card-like display
            with col1:
                if "user_id" in df.columns and not pd.isna(row["user_id"]) and row["user_id"]:
                    st.markdown(f"👤 **User ID:** {row['user_id']}")
                else:
                    st.markdown("👤")  # User icon
            
            with col2:
                if not pd.isna(row['comment']) and row['comment']:  # Only show if not NaN and not empty
                    st.markdown(f"**Comment:** {row['comment']}")
                
                if not pd.isna(row['sentiment']) and row['sentiment']:  # Only show if not NaN
                    sentiment_color = "green" if row['sentiment'].lower() == "positive" else "red"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{row['sentiment']}</span> | **Score:** {row['score']:.2f}", unsafe_allow_html=True)
                
                if "ProblemSummary" in df.columns and not pd.isna(row["ProblemSummary"]) and row["ProblemSummary"]:  # Only show if not NaN and not empty
                    st.markdown(f"**Problem Summary:** {row['ProblemSummary']}")
            
            st.markdown("---")  # Separator for visual appeal
else:
    st.info("No comments yet.")
