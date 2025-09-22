import streamlit as st
import pandas as pd
import pickle
import re
import requests
import base64
import json
from datetime import datetime
import io

# --- Sentilytics Branding ---
brand_color = "#3498db"  # A calming blue
secondary_color = "#2ecc71"  # A vibrant green for success
text_color = "#333333"
bg_color = "#f4f6f7"
input_bg = "#ffffff"
button_color = brand_color

# --- Global Styling ---
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: sans-serif;
}}
h1, h2 {{
    color: {brand_color};
}}
.stTextInput > div > div > input {{
    background-color: {input_bg};
    color: {text_color};
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
}}
.stTextArea > div > div > textarea {{
    background-color: {input_bg};
    color: {text_color};
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
}}
.stButton > button {{
    background-color: {button_color};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
}}
.stButton:hover > button {{
    opacity: 0.8;
}}
.submitted-message {{
    background-color: {secondary_color};
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# --- Page Title & Intro ---
st.title("💬 Sentilytics - Voice of the People")
st.write("Share your thoughts and help us build a better understanding of what matters to everyone.")

# --- Load Models ---
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- GitHub API Setup ---
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
        return pd.DataFrame(columns=["comment", "user_id"]), None

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
        return True
    else:
        st.error(f"Failed to update CSV: {res.text}")
        return False

# --- User Input ---
user_id = st.text_input("Your Username (optional):", value="Anonymous")
user_comment = st.text_area("Share your thoughts:", height=150)

if st.button("Submit"):
    if user_comment.strip() != "":
        df, sha = get_csv()

        new_row = {
            "user_id": user_id,
            "comment": user_comment
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if sha:
            if update_csv(df, sha):
                st.success("Comment uploaded! Thank you for sharing.", text_color=brand_color)
            else:
                st.error("There was an error uploading your comment.")
        else:
            st.error("Could not fetch CSV SHA from GitHub.")
    else:
        st.warning("Please enter a comment before submitting.")
