import streamlit as st
import pandas as pd
from datasets import Dataset
from utils.visualization import plot_distribution, generate_wordcloud

def render(model, tokenizer, trainer):
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
                st.pyplot(plot_distribution(data, difficulty_mapping))

                # Generate and display word cloud with stopwords removed
                st.subheader('Word Cloud of Sentences')
                text = ' '.join(data['sentence'])
                st.pyplot(generate_wordcloud(text))

                st.download_button(label='Download Predictions', data=data.to_csv(index=False).encode('utf-8'),
                                   file_name='predicted_difficulties.csv', mime='text/csv')
        else:
            st.error('Uploaded file does not contain required "sentence" column.')
