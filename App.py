import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import warnings
from pytube import YouTube
import speech_recognition as sr
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
import plotly.express as px
from streamlit_player import st_player
import torch

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

# Suppress specific warnings
warnings.filterwarnings("ignore",
                        message="do_lowercase_and_remove_accent is passed as a keyword argument, but this won't do anything. FlaubertTokenizer will always set it to False.")

# Load the model and tokenizer from Hugging Face Hub
model_name = "mn00/Flaubert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = FlaubertForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = FlaubertTokenizer.from_pretrained(model_name)
    trainer = Trainer(model=model)
    return model, tokenizer, trainer

model, tokenizer, trainer = load_model()

# Print model and tokenizer details
st.write(f"Model: {model}")
st.write(f"Tokenizer: {tokenizer}")

# Difficulty mapping
difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

# Define the Streamlit layout
st.title('Text Difficulty Prediction App')
st.write(
    'This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, record audio, or input long texts such as song lyrics.')

# Tab layout for file upload, text input, long text input, YouTube video input, YouTube videos by difficulty, and audio input
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Upload CSV", "Input Sentence", "Input Long Text", "YouTube Video URL", "YouTube Videos by Difficulty",
     "Record or Upload Audio", "User Feedback"])

def preprocess_and_predict(data):
    data['sentence'] = data['sentence'].astype(str)
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(
        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=512), batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    predictions = trainer.predict(dataset).predictions
    predicted_classes = predictions.argmax(axis=1)
    
    # Diagnostic information
    st.write("Raw predictions:", predictions)
    st.write("Predicted classes:", predicted_classes)
    
    data['difficulty'] = [difficulty_mapping[i] for i in predicted_classes]
    return data

with tab1:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'sentence' in data.columns:
                st.write('Data successfully loaded!')
                with st.spinner('Predicting...'):
                    data = preprocess_and_predict(data)
                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])

                    # Display prediction distribution with Plotly
                    st.subheader('Prediction Distribution')
                    fig = px.histogram(data, x='difficulty', category_orders={'difficulty': list(difficulty_mapping.values())})
                    st.plotly_chart(fig)

                    # Generate and display word cloud with stopwords removed
                    st.subheader('Word Cloud of Sentences')
                    text = ' '.join(data['sentence'])
                    stopwords = set(STOPWORDS).union({"de", "la", "le", "et", "les", "des", "du", "un", "une"})  # Add common French stopwords
                    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()

                    st.download_button(label='Download Predictions', data=data.to_csv(index=False).encode('utf-8'),
                                       file_name='predicted_difficulties.csv', mime='text/csv')
            else:
                st.error('Uploaded file does not contain required "sentence" column.')
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            traceback.print_exc()
    else:
        st.error('Please upload a CSV file.')

with tab2:
    st.header("Input Sentence Directly")
    sentence = st.text_area("Enter the sentence here:")
    if st.button("Predict Difficulty"):
        if sentence:
            try:
                with st.spinner('Predicting...'):
                    # Process the sentence
                    data = preprocess_and_predict(pd.DataFrame({'sentence': [sentence]}))
                    predicted_difficulty = data['difficulty'][0]

                    st.success(f'The predicted difficulty level for the input sentence is: {predicted_difficulty}')
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                traceback.print_exc()
        else:
            st.error("Please enter a sentence for prediction.")

with tab3:
    st.header("Input Long Text")
    long_text = st.text_area("Enter the text here (e.g., song lyrics, paragraphs). ", height=300)
    if st.button("Predict Difficulty for Long Text"):
        if long_text:
            try:
                with st.spinner('Predicting...'):
                    # Split the long text into sentences (assuming sentences are separated by periods)
                    sentences = long_text.split('.')
                    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences
                    data = preprocess_and_predict(pd.DataFrame({'sentence': sentences}))

                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])

                    # Display prediction distribution with Plotly
                    st.subheader('Prediction Distribution')
                    fig = px.histogram(data, x='difficulty', category_orders={'difficulty': list(difficulty_mapping.values())})
                    st.plotly_chart(fig)

                    # Generate and display word cloud with stopwords removed
                    st.subheader('Word Cloud of Sentences')
                    text = ' '.join(data['sentence'])
                    stopwords = set(STOPWORDS).union({"de", "la", "le", "et", "les", "des", "du", "un", "une"})  # Add common French stopwords
                    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                traceback.print_exc()
        else:
            st.error("Please enter some text for prediction.")

with tab4:
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

                    # Transcribe audio to text
                    recognizer = sr.Recognizer()
                    audio_path = "audio.mp4"
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                        transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")

                    st.write("Transcription:")
                    st.write(transcribed_text)

                    # Process the transcribed text
                    data = preprocess_and_predict(pd.DataFrame({'sentence': [transcribed_text]}))
                    predicted_difficulty = data['difficulty'][0]

                    st.success(f'The predicted difficulty level for the transcribed video is: {predicted_difficulty}')
                except Exception as video_error:
                    st.error(f"Error processing video: {video_error}")
                    traceback.print_exc()
        else:
            st.error("Please enter a YouTube URL for prediction.")

with tab5:
    st.header("Find YouTube Videos by Difficulty")
    difficulty_level = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'), index=2)
    if st.button('Search Videos'):
        query = f"French lessons {difficulty_level}"  # Customize query based on difficulty
        videos = search_youtube_videos(query)
        display_video_results(videos)

with tab6:
    st.header("Record or Upload Audio")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if audio_file is not None:
        with st.spinner('Transcribing audio...'):
            try:
                # Save the uploaded audio file
                audio_path = "uploaded_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")

                st.write("Transcription:")
                st.write(transcribed_text)

                with st.spinner('Predicting difficulty...'):
                    data = preprocess_and_predict(pd.DataFrame({'sentence': [transcribed_text]}))
                    predicted_difficulty = data['difficulty'][0]

                    st.success(f'The predicted difficulty level for the transcribed audio is: {predicted_difficulty}')
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                traceback.print_exc()

with tab7:
    st.header("User Feedback")
    feedback_sentence = st.text_area("Enter a sentence and its difficulty level for feedback:")
    feedback_difficulty = st.selectbox("Select the difficulty level:", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    if st.button("Submit Feedback"):
        if feedback_sentence and feedback_difficulty:
            with open("feedback.csv", "a") as f:
                f.write(f"{feedback_sentence},{feedback_difficulty}\n")
            st.success("Thank you for your feedback!")
        else:
            st.error("Please enter a sentence and select a difficulty level.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    traceback.print_exc()
