
import streamlit as st
import pandas as pd
import requests
import base64
import io
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Comment Insights",
    page_icon="💬",
    layout="wide"
)

# GitHub API Setup
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
        return pd.DataFrame(columns=["user_id", "comment", "sentiment", "score", "ProblemSummary"])

# Initialize session state for tracking considered and disapproved comments
if 'considered_comments' not in st.session_state:
    st.session_state.considered_comments = {}
if 'disapproved_comments' not in st.session_state:
    st.session_state.disapproved_comments = {}

# Callback functions
def mark_considered(comment_id):
    # Remove from disapproved if it was there
    if comment_id in st.session_state.disapproved_comments:
        del st.session_state.disapproved_comments[comment_id]
    
    # Mark as considered
    st.session_state.considered_comments[comment_id] = True

def mark_disapproved(comment_id):
    # Remove from considered if it was there
    if comment_id in st.session_state.considered_comments:
        del st.session_state.considered_comments[comment_id]
    
    # Mark as disapproved
    st.session_state.disapproved_comments[comment_id] = True

# Main App
st.title("💬 Comment Insights Dashboard")

# Sidebar for Filtering
st.sidebar.header("🔍 Comment Filters")
sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment", 
    options=['Positive', 'Negative', 'Neutral'],
    default=['Positive', 'Negative', 'Neutral']
)
user_search = st.sidebar.text_input("Search by User ID")

# Sidebar Statistics
st.sidebar.header("📊 Action Statistics")
st.sidebar.metric("Considered Comments", len(st.session_state.considered_comments))
st.sidebar.metric("Disapproved Comments", len(st.session_state.disapproved_comments))

# Refresh Data
col1, col2 = st.columns([3, 1])
with col2:
    refresh_button = st.button("🔄 Refresh Data", use_container_width=True)

if refresh_button:
    st.session_state.df = get_csv()
else:
    if 'df' not in st.session_state:
        st.session_state.df = get_csv()

df = st.session_state.df

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Comments", len(df), help="Total number of comments")

with col2:
    positive_count = len(df[df['sentiment'].str.lower() == 'positive'])
    st.metric("Positive", positive_count, help="Number of positive comments")

with col3:
    negative_count = len(df[df['sentiment'].str.lower() == 'negative'])
    st.metric("Negative", negative_count, help="Number of negative comments")

with col4:
    neutral_count = len(df[df['sentiment'].str.lower() == 'neutral'])
    st.metric("Neutral", neutral_count, help="Number of neutral comments")

# Sentiment Distribution Chart
st.subheader("Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts()
fig = px.pie(
    values=sentiment_counts.values, 
    names=sentiment_counts.index, 
    color_discrete_sequence=['green', 'red', 'gray']
)
st.plotly_chart(fig, use_container_width=True)

# Apply Filters
filtered_df = df[
    df['sentiment'].str.title().isin(sentiment_filter) & 
    (df['user_id'].str.contains(user_search, case=False) if user_search else True)
]

# Comments Section
st.subheader("Recent Comments")

# Pagination
comments_per_page = 10
total_comments = len(filtered_df)
total_pages = (total_comments + comments_per_page - 1) // comments_per_page

# Page selector
page_number = st.number_input(
    "Select Page", 
    min_value=1, 
    max_value=total_pages, 
    value=1
)

# Calculate start and end indices
start_idx = (page_number - 1) * comments_per_page
end_idx = start_idx + comments_per_page

# Display Comments
for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
    # Determine sentiment color
    sentiment_color = (
        "green" if row['sentiment'].lower() == 'positive' 
        else "red" if row['sentiment'].lower() == 'negative' 
        else "gray"
    )
    
    # Create expander for each comment
    with st.expander(f"Comment by {row['user_id']}"):
        # Comment details
        st.write(f"**Comment:** {row['comment']}")
        
        # Sentiment chip
        st.markdown(f"**Sentiment:** :{sentiment_color}[{row['sentiment']}]")
        
        # Sentiment score
        st.write(f"**Sentiment Score:** {row['score']:.2f}")
        
        # Problem Summary (if available)
        if not pd.isna(row.get('ProblemSummary', '')):
            st.write(f"**Problem Summary:** {row['ProblemSummary']}")
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Considered Button
            considered_key = f"considered_{idx}"
            is_considered = st.session_state.considered_comments.get(idx, False)
            is_disapproved = st.session_state.disapproved_comments.get(idx, False)
            
            if not is_considered and not is_disapproved:
                considered_button = st.button(
                    "🤔 Considered", 
                    key=considered_key,
                    on_click=mark_considered,
                    args=(idx,),
                    use_container_width=True
                )
            elif is_considered:
                st.success("✅ Considered")
        
        with col2:
            # Disapproved Button
            disapproved_key = f"disapproved_{idx}"
            
            if not is_considered and not is_disapproved:
                disapproved_button = st.button(
                    "❌ Disapproved", 
                    key=disapproved_key,
                    on_click=mark_disapproved,
                    args=(idx,),
                    use_container_width=True
                )
            elif is_disapproved:
                st.error("❌ Disapproved")

# Pagination info
st.write(f"Page {page_number} of {total_pages} | Showing {min(comments_per_page, len(filtered_df[start_idx:end_idx]))} of {total_comments} comments")

- Using proper callback mechanisms
- Avoiding direct session state modifications
- Providing a clean, intuitive user interface for comment actions

The code maintains all previous functionality while addressing the specific Streamlit error you encountered.
