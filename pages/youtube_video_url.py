import streamlit as st
from pytube import YouTube
import speech_recognition as sr
import ffmpeg
import os
import pandas as pd
from datasets import Dataset
from utils.visualization import plot_distribution, generate_wordcloud

def render(model, tokenizer, trainer):
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
