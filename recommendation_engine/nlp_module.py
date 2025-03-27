from transformers import pipeline

# Here we use "distilbert-base-uncased-finetuned-sst-2-english" which is fine-tuned on sentiment analysis.
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_text(text):
    """Classify text as positive or negative sentiment"""
    result = classifier(text)
    return result[0]['label'] == 'POSITIVE'


"""
# TESTING
#results = classifier(texts)
# Example texts representing employee messages about shift cancellation
texts = [

    "I'm really sorry, but I won't be able to come in tomorrow.",
    "Looking forward to my shift today, can't wait!",
    "Due to a personal emergency, I have to cancel my shift.",
    "I will definitely be there for my shift.",
    "I cannot make it today",
    "All good, I'll be there",
    "Due to unforeseen circumstances, I am cancelling my shift",
    "I will be coming during my shift",
    "I have a meeting, so I won't be available",
    "I have a doctor's appointment, so I won't be able to make it"
]

## Print the results
#for text, result in zip(texts, results):
#    print(f"Text: {text}")
#    print(f"Prediction: {result['label']} with score {result['score']:.4f}\n")
"""