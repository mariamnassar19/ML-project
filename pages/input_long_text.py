import streamlit as st
import pandas as pd
from datasets import Dataset
from utils.visualization import plot_distribution, generate_wordcloud

def render(model, tokenizer, trainer):
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
                st.pyplot(plot_distribution(data, difficulty_mapping))

                # Generate and display word cloud with stopwords removed
                st.subheader('Word Cloud of Sentences')
                text = ' '.join(data['sentence'])
                st.pyplot(generate_wordcloud(text))
