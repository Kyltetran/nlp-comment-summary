import sqlite_patch  # 👈 This must come first
import os
import json
import time
import streamlit as st
from youtube_summary_tool_copy import (
    analyze_youtube_comments,
    answer_question,
    extract_video_id,
    CURRENT_VIDEO_ID,
    close_chroma_connection
)


# Set Streamlit page config
st.set_page_config(page_title="YouTube Comment Analyzer", layout="centered")

# Session state for persistent results
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None

# Sidebar: Status section
st.sidebar.title("📊 Status")
status = {
    'database_exists': os.path.exists("chroma"),
    'current_video_id': CURRENT_VIDEO_ID
}

metadata_path = None
if CURRENT_VIDEO_ID:
    metadata_path = os.path.join(
        "chroma", CURRENT_VIDEO_ID, "video_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status['metadata'] = metadata
        except Exception as e:
            status['metadata_error'] = str(e)

st.sidebar.write("*Database exists:*", status['database_exists'])
st.sidebar.write("*Current video ID:*", status['current_video_id'])
if 'metadata' in status:
    st.sidebar.write("*Metadata:*")
    st.sidebar.json(status['metadata'])

# App title
st.title("🎥 YouTube Comment Analyzer")

# Step 1: Input YouTube URL
st.header("1️⃣ Analyze a YouTube Video")
youtube_url = st.text_input("Enter the YouTube video URL:")

if st.button("Analyze Comments"):
    if not youtube_url:
        st.error("Please enter a YouTube URL.")
    else:
        with st.spinner("Analyzing comments..."):
            try:
                close_chroma_connection()
                result = analyze_youtube_comments(youtube_url)
                st.session_state.latest_results = result
                st.success("Analysis complete!")
                st.json(result)
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Step 2: Ask a question
st.header("2️⃣ Ask a Question")
question = st.text_input("Ask a question based on the comments:")
k_value = st.number_input(
    "Optional: number of comments to use (k)", min_value=1, step=1)

if st.button("Get Answer"):
    if not question:
        st.error("Please enter a question.")
    elif not os.path.exists("chroma"):
        st.error("No comment database found. Please analyze a video first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                k = int(k_value) if k_value else None
                result = answer_question(question, k=k)
                st.success("Answer generated!")
                st.markdown("### ✅ Answer")
                st.write(result["answer"])
                st.caption(
                    f"Used {result['k_used']} comments | Time: {result['processing_time']}")
            except Exception as e:
                st.error(f"Error while answering: {e}")
