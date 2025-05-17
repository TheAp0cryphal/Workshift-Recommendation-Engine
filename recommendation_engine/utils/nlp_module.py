from transformers import pipeline

# Here we use "distilbert-base-uncased-finetuned-sst-2-english" which is fine-tuned on sentiment analysis.
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_text(text):
    """Classify text as positive or negative sentiment"""
    result = classifier(text)
    return result[0]['label'] == 'POSITIVE'