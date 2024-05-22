import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
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
    """Search videos on YouTube based on the query."""
    response = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video'
    ).execute()
    return response

def display_video_results(videos):
    """Display video results in Streamlit."""
    for item in videos['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_description = item['snippet']['description']
        video_link = f"https://www.youtube.com/watch?v={video_id}"

        st.write(f"### {video_title}")
        st.write(f"Description: {video_description[:200]}...")  # Show first 200 characters
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
    audio = AudioSegment.from_file(audio_path)
    wav_path = "converted_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def plot_difficulty_distribution(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data['difficulty'], bins=len(set(data['difficulty'])), color='skyblue', alpha=0.7)
    plt.title('Distribution of Sentence Difficulty')
    plt.xlabel('Difficulty')
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot(plt)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="do_lowercase_and_remove_accent is passed as a keyword argument, but this won't do anything. FlaubertTokenizer will always set it to False.")

# Load the model and tokenizer from Hugging Face Hub
model_name = "shiqi-017/flaubert"
difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

try:
    model, tokenizer, trainer = load_model_and_tokenizer(model_name)

    # Define the Streamlit layout
    st.title('Text Difficulty Prediction App')
    st.write('This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, or provide a YouTube video URL.')

    # Tab layout for file upload, text input, YouTube video input, YouTube videos by difficulty, and feedback
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload CSV", "Input Sentence", "YouTube Video URL", "YouTube Videos by Difficulty", "Feedback"])

    with tab1:
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                if 'sentence' in data.columns:
                    st.write('Data successfully loaded!')
                    with st.spinner('Predicting...'):
                        data['sentence'] = data['sentence'].astype(str)
                        predicted_classes = predict_difficulty(trainer, tokenizer, data['sentence'])
                        data['difficulty'] = [difficulty_mapping[i] for i in predicted_classes]
                        st.write('Predictions complete!')
                        st.dataframe(data[['sentence', 'difficulty']])
                        plot_difficulty_distribution(data)
                        st.download_button(label='Download Predictions', data=data.to_csv(index=False).encode('utf-8'), file_name='predicted_difficulties.csv', mime='text/csv')
                else:
                    st.error('Uploaded file does not contain required "sentence" column.')
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                traceback.print_exc()

    with tab2:
        st.header("Input Sentence Directly")
        sentence = st.text_area("Enter the sentence here:")
        if st.button("Predict Difficulty"):
            if sentence:
                with st.spinner('Predicting...'):
                    try:
                        predicted_classes = predict_difficulty(trainer, tokenizer, [sentence])
                        predicted_difficulty = difficulty_mapping[predicted_classes[0]]
                        st.success(f'The predicted difficulty level for the input sentence is: {predicted_difficulty}')
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
            else:
                st.error("Please enter a sentence for prediction.")

    with tab3:
        st.header("YouTube Video URL")
        youtube_url = st.text_input("Enter YouTube URL here:")
        if st.button("Predict Difficulty from Video"):
            if youtube_url:
                with st.spinner('Processing YouTube video...'):
                    try:
                        # Download the YouTube video
                        yt = YouTube(youtube_url)
                        audio_stream = yt.streams.filter(only_audio=True).first()
                        audio_file = audio_stream.download(filename="audio.mp4")
                        # Convert audio to WAV format
                        wav_path = convert_audio_to_wav(audio_file)
                        # Transcribe audio to text
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio_data = recognizer.record(source)
                            transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")
                        st.write("Transcription:")
                        st.write(transcribed_text)
                        # Process the transcribed text
                        predicted_classes = predict_difficulty(trainer, tokenizer, [transcribed_text])
                        predicted_difficulty = difficulty_mapping[predicted_classes[0]]
                        st.success(f'The predicted difficulty level for the transcribed video is: {predicted_difficulty}')
                    except Exception as video_error:
                        st.error(f"Error processing video: {video_error}")
                        traceback.print_exc()
                    finally:
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        if 'wav_path' in locals() and os.path.exists(wav_path):
                            os.remove(wav_path)
            else:
                st.error("Please enter a YouTube URL for prediction.")

    with tab4:
        st.header("Find YouTube Videos by Difficulty")
        difficulty_level = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'), index=2)
        if st.button('Search Videos'):
            query = f"French lessons {difficulty_level}"  # Customize query based on difficulty
            videos = search_youtube_videos(query)
            display_video_results(videos)

    with tab5:
        st.header("We value your feedback!")
        feedback_text = st.text_area("Please share your experience with us:")
        feedback_email = st.text_input("Email (optional):")
        feedback_button = st.button("Submit Feedback")
        if feedback_button and feedback_text:
            # Here you would handle saving the feedback to a file or database
            st.success("Thank you for your feedback!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    traceback.print_exc()
