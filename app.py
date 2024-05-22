import streamlit as st
from utils.model import load_model, predict
from pages import upload_csv, input_sentence, input_long_text, youtube_video_url, youtube_videos_by_difficulty, feedback

# Load the model and tokenizer
model, tokenizer, trainer = load_model()

# Define the Streamlit layout
st.title('Text Difficulty Prediction App')
st.write(
    'This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, or input long texts such as song lyrics.')

# Tab layout for file upload, text input, long text input, YouTube video input, YouTube videos by difficulty, and feedback
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Upload CSV", "Input Sentence", "Input Long Text", "YouTube Video URL", "YouTube Videos by Difficulty", "Feedback"])

with tab1:
    upload_csv.render(model, tokenizer, trainer)

with tab2:
    input_sentence.render(model, tokenizer, trainer)

with tab3:
    input_long_text.render(model, tokenizer, trainer)

with tab4:
    youtube_video_url.render(model, tokenizer, trainer)

with tab5:
    youtube_videos_by_difficulty.render()

with tab6:
    feedback.render()
