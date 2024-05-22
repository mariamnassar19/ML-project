import streamlit as st
from utils.youtube import search_youtube_videos, display_video_results

def render():
    st.header("Find YouTube Videos by Difficulty")
    difficulty_level = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'), index=2)
    if st.button('Search Videos'):
        query = f"French lessons {difficulty_level}"  # Customize query based on difficulty
        videos = search_youtube_videos(query)
        display_video_results(videos)
