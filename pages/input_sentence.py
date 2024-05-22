import streamlit as st
from datasets import Dataset

def render(model, tokenizer, trainer):
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
