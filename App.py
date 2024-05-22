import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import traceback
import warnings
from pytube import YouTube
import speech_recognition as sr
from pydub import AudioSegment
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from streamlit_player import st_player

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')
if not api_key:
    st.error("API Key not found. Make sure the environment variable YOUTUBE_API_KEY is set.")
    st.stop()

# Initialize the YouTube client
youtube = build('youtube', 'v3', developerKey=api_key)

def search_youtube_videos(query, max_results=10):
    response = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video'
    ).execute()
    return response

def display_video_results(videos):
    for item in videos['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_description = item['snippet']['description']
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        st.write(f"### {video_title}")
        st.write(f"Description: {video_description[:200]}...")
        st_player(video_link)

@st.cache
def load_model_and_tokenizer(model_name="mn00/Flaubertmodel"):
    model = FlaubertForSequenceClassification.from_pretrained(model_name)
    tokenizer = FlaubertTokenizer.from_pretrained(model_name)
    trainer = Trainer(model=model)
    return model, tokenizer, trainer

def predict_difficulty(trainer, tokenizer, sentences):
    data = pd.DataFrame({'sentence': sentences})
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=512), batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    predictions = trainer.predict(dataset).predictions
    predicted_classes = predictions.argmax(axis=1)
    return predicted_classes

def convert_audio_to_wav(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = "converted_audio.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error("Failed to convert audio to WAV format.")
        raise e

def save_feedback(feedback_text, feedback_email):
    with open("feedback.txt", "a") as file:
        file.write(f"Email: {feedback_email if feedback_email else 'N/A'}, Feedback: {feedback_text}\n")

model_name = "shiqi-017/flaubert"
difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

try:
    model, tokenizer, trainer = load_model_and_tokenizer(model_name)
    st.title('Text Difficulty Prediction App')
    st.write('This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, or give feedback.')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload CSV", "Input Sentence", "YouTube Video URL", "YouTube Videos by Difficulty", "Feedback"])

    with tab5:
        st.header("Feedback")
        feedback_text = st.text_area("Share your feedback:")
        feedback_email = st.text_input("Email (optional):")
        if st.button("Submit Feedback"):
            if feedback_text:
                save_feedback(feedback_text, feedback_email)
                st.success("Thank you for your feedback!")
            else:
                st.error("Please enter some feedback before submitting.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    traceback.print_exc()
