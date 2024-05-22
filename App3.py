import os
import streamlit as st
import pandas as pd
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import traceback
import warnings
from googleapiclient.discovery import build
from nltk.corpus import stopwords
import nltk
import speech_recognition as sr
import ffmpeg

# Download the stopwords from nltk
nltk.download('stopwords')

# Retrieve API key from Streamlit secrets
youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

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

# Load the model and tokenizer
model_path = '/Users/mariamnassar/Documents/Semester 2/Data science and machine learning/project model/content/flaubert_finetuned_full'

try:
    model = FlaubertForSequenceClassification.from_pretrained(model_path)
    tokenizer = FlaubertTokenizer.from_pretrained(model_path)
    trainer = Trainer(model=model)

    # Difficulty mapping
    difficulty_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

    # Initialize an empty DataFrame to store feedback
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = pd.DataFrame(columns=['sentence', 'difficulty'])

    # Define the Streamlit layout
    st.title('Text Difficulty Prediction App')
    st.write(
        'This application predicts the difficulty level of French sentences. You can upload a CSV file, input sentences directly, provide a YouTube video URL, or input long texts such as song lyrics.')

    # Tab layout for file upload, text input, long text input, YouTube video input, YouTube videos by difficulty, and feedback
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Upload CSV", "Input Sentence", "Input Long Text", "YouTube Video URL", "YouTube Videos by Difficulty", "Feedback"])

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
        long_text = st.text_area("Enter the text here (e.g., song lyrics, paragraphs):", height=300)
        if st.button("Predict Difficulty for Long Text"):
            if long_text:
                with st.spinner('Predicting...'):
                    # Split the long text into sentences (assuming sentences are separated by periods)
                    sentences = long_text.split('.')
                    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences

                    # Process and predict each sentence
                    data = pd.DataFrame({'sentence': sentences})
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

                        # Convert audio to WAV format using ffmpeg
                        wav_file = "audio.wav"
                        ffmpeg.input(audio_file).output(wav_file).run(overwrite_output=True)
                        os.remove(audio_file)  # Remove original audio file

                        # Transcribe audio to text
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_file) as source:
                            audio_data = recognizer.record(source)
                            transcribed_text = recognizer.recognize_google(audio_data, language="fr-FR")
                        os.remove(wav_file)  # Remove WAV file after transcription

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
        st.header("Feedback")
        feedback_sentence = st.text_area("Please provide a sentence for feedback:")
        feedback_difficulty = st.selectbox("Select the difficulty level:", ('A1', 'A2', 'B1', 'B2', 'C1', 'C2'))

        if st.button("Submit Feedback"):
            if feedback_sentence:
                # Add feedback to session state DataFrame
                new_feedback = pd.DataFrame([[feedback_sentence, feedback_difficulty]], columns=['sentence', 'difficulty'])
                st.session_state.feedback_data = pd.concat([st.session_state.feedback_data, new_feedback], ignore_index=True)

                st.success("Thank you for your feedback!")
                st.dataframe(st.session_state.feedback_data)
                
                # Option to download feedback
                st.download_button(
                    label="Download Feedback Data",
                    data=st.session_state.feedback_data.to_csv(index=False).encode('utf-8'),
                    file_name='feedback_data.csv',
                    mime='text/csv'
                )
            else:
                st.error("Please enter a sentence for feedback before submitting.")

except Exception as e:
    st.error("An error occurred: {}".format(str(e)))
    traceback.print_exc()
