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
import subprocess

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')
if not api_key:
    st.error("API Key not found. Make sure the environment variable YOUTUBE_API_KEY is set.")
    st.stop()

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

if not check_ffmpeg():
    st.error("FFmpeg is not installed or not found. Please install FFmpeg to process videos.")
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

@st.cache_resource
def load_model_and_tokenizer(model_name="mn00/trial"):
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

# Suppress specific warnings
warnings.filterwarnings("ignore", message="do_lowercase_and_remove_accent is passed as a keyword argument, but this won't do anything. FlaubertTokenizer will always set it to False.")

model_name = "shiqi-017/flaubert"
difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

try:
    model, tokenizer, trainer = load_model_and_tokenizer(model_name)
    st.title('Text Difficulty Prediction App')
    st.write('This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, or provide a YouTube video URL.')

    tab1, tab2, tab3, tab4 = st.tabs(["Upload CSV", "Input Sentence", "YouTube Video URL", "YouTube Videos by Difficulty"])

    with tab1:
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'sentence' in data.columns:
                st.write('Data successfully loaded!')
                with st.spinner('Predicting...'):
                    data['sentence'] = data['sentence'].astype(str)
                    predicted_classes = predict_difficulty(trainer, tokenizer, data['sentence'])
                    data['difficulty'] = [difficulty_mapping[i] for i in predicted_classes]
                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])
            else:
                st.error('Uploaded file does not contain required "sentence" column.')

    with tab2:
        st.header("Input Sentence Directly")
        sentence = st.text_area("Enter the sentence here:")
        if sentence and st.button("Predict Difficulty"):
            with st.spinner('Predicting...'):
                predicted_classes = predict_difficulty(trainer, tokenizer, [sentence])
                predicted_difficulty = difficulty_mapping[predicted_classes[0]]
                st.success(f'The predicted difficulty level for the input sentence is: {predicted_difficulty}')

    with tab3:
        st.header("YouTube Video URL")
        youtube_url = st.text_input("Enter YouTube URL here:")
        if youtube_url and st.button("Predict Difficulty from Video"):
            with st.spinner('Processing YouTube video...'):
                yt = YouTube(youtube_url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                audio_file = audio_stream.download(filename="audio.mp4")
                wav_path = convert_audio_to_wav(audio_file)
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")
                st.write("Transcription:")
                st.write(transcribed_text)
                predicted_classes = predict_difficulty(trainer, tokenizer, [transcribed_text])
                predicted_difficulty = difficulty_mapping[predicted_classes[0]]
                st.success(f'The predicted difficulty level for the transcribed video is: {predicted_difficulty}')

    with tab4:
        st.header("Find YouTube Videos by Difficulty")
        difficulty_level = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'))
        if st.button('Search Videos'):
            query = f"French lessons {difficulty_level}"
            videos = search_youtube_videos(query)
            display_video_results(videos)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    traceback.print_exc()
