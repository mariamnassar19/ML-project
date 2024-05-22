# test_transformers.py
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

model_name = "mn00/trial"

try:
    model = FlaubertForSequenceClassification.from_pretrained(model_name)
    tokenizer = FlaubertTokenizer.from_pretrained(model_name)
    print("Transformers library and model loaded successfully.")
except Exception as e:
    print(f"Error loading transformers library: {e}")

