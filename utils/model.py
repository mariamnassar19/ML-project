import os
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer

def load_model():
    model_path = '/Users/mariamnassar/Documents/Semester 2/Data science and machine learning/project model/content/flaubert_finetuned_full'
    model = FlaubertForSequenceClassification.from_pretrained(model_path)
    tokenizer = FlaubertTokenizer.from_pretrained(model_path)
    trainer = Trainer(model=model)
    return model, tokenizer, trainer

def predict(trainer, tokenizer, sentences, max_length=512):
    dataset = Dataset.from_pandas(pd.DataFrame({'sentence': sentences}))
    dataset = dataset.map(
        lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=max_length),
        batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    predictions = trainer.predict(dataset).predictions
    predicted_classes = predictions.argmax(axis=1)
    return predicted_classes
