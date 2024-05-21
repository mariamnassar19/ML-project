import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import traceback
import warnings
from pytube import YouTube
import whisper
import os
from googleapiclient.discovery import build
from nltk.corpus import stopwords
import nltk
import torch

# Download the stopwords from nltk
nltk.download('stopwords')

# Retrieve API key from Streamlit secrets
try:
    api_key = st.secrets["YOUTUBE_API_KEY"]
except KeyError:
    st.error("API Key not found. Make sure the environment variable YOUTUBE_API_KEY is set in Streamlit secrets.")
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
        st.markdown(f"[Watch Video]({video_link})")

# Suppress specific warnings
warnings.filterwarnings("ignore",
                        message="do_lowercase_and_remove_accent is passed as a keyword argument, but this won't do anything. FlaubertTokenizer will always set it to False.")

# Load the model and tokenizer from Hugging Face Hub
model_name = "mn00/Flaubert"  # Replace with your model name on Hugging Face Hub

try:
    model = FlaubertForSequenceClassification.from_pretrained(model_name)
    tokenizer = FlaubertTokenizer.from_pretrained(model_name)
    trainer = Trainer(model=model)
    whisper_model = whisper.load_model("base")

    # Difficulty mapping
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Define the Streamlit layout
    st.title('Text Difficulty Prediction App')
    st.write(
        'This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, record audio, or input long texts such as song lyrics.')

    # Tab layout for file upload, text input, long text input, YouTube video input, YouTube videos by difficulty, and audio input
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Upload CSV", "Input Sentence", "Input Long Text", "YouTube Video URL", "YouTube Videos by Difficulty", "Record or Upload Audio"])

    with tab1:
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'sentence' in data.columns:
                st.write('Data successfully loaded!')
                with st.spinner('Predicting...'):
                    # Processing and prediction
                    data['sentence'] = data['sentence'].astype(str)
                    dataset = Dataset.from_pandas(data)
                    dataset = dataset.map(
                        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True,
                                                   max_length=512), batched=True)
                    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                    predictions = trainer.predict(dataset).predictions
                    predicted_classes = predictions.argmax(axis=1)
                    data['difficulty'] = [difficulty_mapping[i] for i in predicted_classes]

                    st.write('Predictions complete!')
                    st.dataframe(data[['sentence', 'difficulty']])

                    # Display prediction distribution
                    st.subheader('Prediction Distribution')
                    fig, ax = plt.subplots()
                    sns.countplot(x='difficulty', data=data, order=difficulty_mapping.values(), ax=ax)
                    st.pyplot(fig)

                    # Generate and display word cloud with stopwords removed
                    st.subheader('Word Cloud of Sentences')
                    text = ' '.join(data['sentence'])
                    french_stopwords = set(stopwords.words('french'))
                    wordcloud = WordCloud(width=800, height=400, background_color='white',
                                          stopwords=french_stopwords).generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()

                    st.download_button(label='Download Predictions', data=data.to_csv(index=False).encode('utf-8'),
                                       file_name='predicted_difficulties.csv', mime='text/csv')
            else:
                st.error('Uploaded file does not contain required "sentence" column.')

    with tab2:
        st.header("Input Sentence Directly")
        sentence = st.text_area("Enter the sentence here:")
        if st.button("Predict Difficulty"):
            if sentence:
                with st.spinner('Predicting...'):
                    # Process the sentence
                    dataset = Dataset.from_pandas(pd.DataFrame({'sentence': [sentence]}))
                    dataset = dataset.map(
                        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True,
                                                   max_length=512), batched=True)
                    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
                    predictions = trainer.predict(dataset).predictions
                    predicted_class = predictions.argmax(axis=1)
                    predicted_difficulty = difficulty_mapping[predicted_class[0]]

                    st.success(f'The predicted difficulty level for the input sentence is: {predicted_difficulty}')
            else:
                st.error("Please enter a sentence for prediction.")

    with tab3:
        st.header("Input Long Text")
        long_text = st.text_area("Enter the text here (e.g., song lyrics, paragraphs). ", height=300)
        if st.button("Predict Difficulty for Long Text"):
            if long_text:
                with st.spinner('Predicting...'):
                    # Split the long text into sentences (assuming sentences are separated by periods)
                    sentences = long

