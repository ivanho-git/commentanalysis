import streamlit as st
import pandas as pd

st.title("📝 Comment Output Page")

csv_file = "comments.csv"

# Load comments
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["comment", "sentiment", "score"])

# Display comments in card-like layout
if not df.empty:
    # Show latest comments first
    for idx, row in df[::-1].iterrows():
        with st.container():
            # Optional: horizontal layout
            col1, col2 = st.columns([1, 5])
            
            # Column 1: User icon or placeholder
            col1.markdown("👤")  # You can replace with actual user if available
            
            # Column 2: Comment content
            col2.markdown(f"**Comment:** {row['comment']}")
            col2.markdown(f"**Sentiment:** {row['sentiment']}  |  **Score:** {row['score']:.2f}")
            
            st.markdown("---")  # Separator between comments
else:
    st.info("No comments yet.")
