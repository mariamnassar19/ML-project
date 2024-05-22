import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import traceback
import warnings
from pytube import YouTube
import speech_recognition as sr
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
import nltk

# Load environment variables
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

if not api_key:
    raise ValueError("API Key not found. Make sure the environment variable YOUTUBE_API_KEY is set.")

# Initialize the YouTube client
youtube = build('youtube', 'v3', developerKey=api_key)

# Download and cache French stopwords
@st.cache(show_spinner=False)
def load_stopwords():
    nltk.download('stopwords')
    return set(nltk.corpus.stopwords.words('french'))

french_stopwords = load_stopwords()

# Suppress specific warnings
warnings.filterwarnings("ignore", message="do_lowercase_and_remove_accent is passed as a keyword argument, but this won't do anything. FlaubertTokenizer will always set it to False.")

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'mn00/Flaubert'
    model = FlaubertForSequenceClassification.from_pretrained(model_path)
    tokenizer = FlaubertTokenizer.from_pretrained(model_path)
    trainer = Trainer(model=model)
    return model, tokenizer, trainer

model, tokenizer, trainer = load_model()

# Difficulty mapping
difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

# Define the Streamlit layout
st.title('Text Difficulty Prediction App')
st.write('This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, record audio, or input long texts such as song lyrics.')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Upload CSV", "Input Sentence", "Input Long Text", "YouTube Video URL", "YouTube Videos by Difficulty", "Record or Upload Audio"])

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
                    dataset = Dataset.from_pandas(data)
                    dataset = dataset.map(
                        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=512), batched=True)
                    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                    predictions = trainer.predict(dataset).predictions
                    data['difficulty'] = [difficulty_mapping[predictions.argmax(axis=1)[i]] for i in range(len(predictions))]

                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])

                    st.subheader('Word Cloud of Sentences')
                    text = ' '.join(data['sentence'])
                    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=french_stopwords).generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()

                    st.download_button('Download Predictions', data.to_csv(index=False).encode('utf-8'), file_name='predicted_difficulties.csv', mime='text/csv')
            else:
                st.error('Uploaded file does not contain required "sentence" column.')
        except Exception as e:
            st.error(f"Error processing file: {e}")
            traceback.print_exc()

with tab2:
    st.header("Input Sentence Directly")
    sentence = st.text_area("Enter the sentence here:")
    if st.button("Predict Difficulty"):
        if sentence:
            try:
                with st.spinner('Predicting...'):
                    # Process the sentence
                    dataset = Dataset.from_pandas(pd.DataFrame({'sentence': [sentence]}))
                    dataset = dataset.map(
                        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=512), batched=True)
                    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                    predictions = trainer.predict(dataset).predictions
                    predicted_class = predictions.argmax(axis=1)
                    predicted_difficulty = difficulty_mapping[predicted_class[0]]

                    st.success(f'The predicted difficulty level for the input sentence is: {predicted_difficulty}')
            except Exception as e:
                st.error(f"Error predicting difficulty: {e}")
                traceback.print_exc()
        else:
            st.error("Please enter a sentence for prediction.")

with tab3:
    st.header("Input Long Text")
    long_text = st.text_area("Enter the text here (e.g., song lyrics, paragraphs):", height=300)
    if st.button("Predict Difficulty for Long Text"):
        if long_text:
            try:
                with st.spinner('Predicting...'):
                    # Split the long text into sentences (assuming sentences are separated by periods)
                    sentences = long_text.split('.')
                    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences
                    data = pd.DataFrame({'sentence': sentences})
                    dataset = Dataset.from_pandas(data)
                    dataset = dataset.map(
                        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True,
                                                   max_length=512), batched=True)
                    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                    predictions = trainer.predict(dataset).predictions
                    data['difficulty'] = [difficulty_mapping[predictions.argmax(axis=1)[i]] for i in range(len(predictions))]

                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])
            except Exception as e:
                st.error(f"Error predicting difficulty: {e}")
                traceback.print_exc()
        else:
            st.error("Please enter some text for prediction.")

with tab4:
    st.header("YouTube Video URL")
    youtube_url = st.text_input("Enter YouTube URL here:")
    if st.button("Predict Difficulty from Video"):
        if youtube_url:
            try:
                with st.spinner('Processing YouTube video...'):
                    yt = YouTube(youtube_url)
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    if audio_stream:
                        audio_file = audio_stream.download(filename="audio.mp4")

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(audio_file) as source:
                            audio_data = recognizer.record(source)
                            transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")
                        os.remove(audio_file)  # Remove audio file after transcription

                        st.write("Transcription:")
                        st.write(transcribed_text)

                        # Process the transcribed text
                        dataset = Dataset.from_pandas(pd.DataFrame({'sentence': [transcribed_text]}))
                        dataset = dataset.map(
                            lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True,
                                                       max_length=512), batched=True)
                        dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                        predictions = trainer.predict(dataset).predictions
                        predicted_class = predictions.argmax(axis=1)
                        predicted_difficulty = difficulty_mapping[predicted_class[0]]

                        st.success(f'The predicted difficulty level for the transcribed video is: {predicted_difficulty}')
                    else:
                        st.error("No audio stream available in this video.")
            except Exception as e:
                st.error(f"Error processing video: {e}")
                traceback.print_exc()
        else:
            st.error("Please enter a YouTube URL for prediction.")

with tab5:
    st.header("Find YouTube Videos by Difficulty")
    difficulty_level = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'), index=2)
    if st.button('Search Videos'):
        try:
            videos = search_youtube_videos(f"French lessons {difficulty_level}")
            display_video_results(videos)
        except Exception as e:
            st.error(f"Error searching videos: {e}")
            traceback.print_exc()

with tab6:
    st.header("Record or Upload Audio")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if audio_file is not None:
        try:
            with st.spinner('Transcribing audio...'):
                # Save the uploaded audio file
                audio_path = "uploaded_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")
                os.remove(audio_path)  # Remove audio file after transcription

                st.write("Transcription:")
                st.write(transcribed_text)

                # Process the transcribed text
                dataset = Dataset.from_pandas(pd.DataFrame({'sentence': [transcribed_text]}))
                dataset = dataset.map(
                    lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True,
                                               max_length=512), batched=True)
                dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                predictions = trainer.predict(dataset).predictions
                predicted_class = predictions.argmax(axis=1)
                predicted_difficulty = difficulty_mapping[predicted_class[0]]

                st.success(f'The predicted difficulty level for the transcribed audio is: {predicted_difficulty}')
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            traceback.print_exc()

# Utility functions
def search_youtube_videos(query, max_results=10):
    """Search videos on YouTube based on the query."""
    try:
        response = youtube.search().list(q=query, part='snippet', maxResults=max_results, type='video').execute()
        return response
    except Exception as e:
        st.error(f"Failed to search YouTube videos: {e}")
        return None

def display_video_results(videos):
    """Display video results in Streamlit."""
    if videos:
        for item in videos['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            video_description = item['snippet']['description']
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            st.write(f"### {video_title}")
            st.write(f"Description: {video_description[:200]}...")
            st.markdown(f"[Watch Video]({video_link})")
