import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer, AddedToken
from datasets import Dataset
import traceback
import warnings
from pytube import YouTube
import speech_recognition as sr
from pydub import AudioSegment
import os
from googleapiclient.discovery import build
from streamlit_player import st_player

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["YOUTUBE_API_KEY"]
if not api_key:
    st.error("API Key not found. Make sure the environment variable YOUTUBE_API_KEY is set in Streamlit secrets.")
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

# Define a custom hash function for AddedToken
def hash_added_token(token):
    return hash((token.content, token.single_word, token.lstrip, token.rstrip, token.normalized, token.special))

@st.cache(hash_funcs={AddedToken: hash_added_token})
def load_model_and_tokenizer(model_name="shiqi-017/flaubert"):
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

    with tab1:
        st.header("Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())
            sentences = data['sentence'].tolist()
            predictions = predict_difficulty(trainer, tokenizer, sentences)
            data['difficulty'] = [difficulty_mapping[p] for p in predictions]
            st.write(data)

    with tab2:
        st.header("Input Sentence")
        sentence = st.text_area("Enter a French sentence")
        if st.button("Predict Difficulty"):
            if sentence:
                sentences = [sentence]
                predictions = predict_difficulty(trainer, tokenizer, sentences)
                st.write(f"The predicted difficulty is: {difficulty_mapping[predictions[0]]}")
            else:
                st.error("Please enter a sentence")

    with tab3:
        st.header("YouTube Video URL")
        video_url = st.text_input("Enter YouTube video URL")
        if st.button("Extract and Predict"):
            if video_url:
                video = YouTube(video_url)
                video_stream = video.streams.filter(only_audio=True).first()
                audio_path = video_stream.download(filename="video_audio.mp4")
                wav_path = convert_audio_to_wav(audio_path)
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language="fr-FR")
                    sentences = [text]
                    predictions = predict_difficulty(trainer, tokenizer, sentences)
                    st.write(f"Transcribed text: {text}")
                    st.write(f"The predicted difficulty is: {difficulty_mapping[predictions[0]]}")
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand the audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
            else:
                st.error("Please enter a valid YouTube video URL")

    with tab4:
        st.header("YouTube Videos by Difficulty")
        search_query = st.text_input("Enter search query")
        if st.button("Search Videos"):
            if search_query:
                try:
                    videos = search_youtube_videos(search_query)
                    display_video_results(videos)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please enter a search query")

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
