# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch

summariser_tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
summariser_model = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")

def summarise(test):
    inputs = summariser_tokenizer(test, max_length=1024, truncation=True, return_tensors="pt")

    # Step 2: Generate the summary
    summary_ids = summariser_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Step 3: Decode the generated summary
    summary = summariser_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Print the summary
    return summary

sentiment_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

def analyze_sentiment(input_text):

    # Step 1: Tokenize the input text
    inputs = sentiment_tokenizer(input_text, return_tensors="pt")
    
    # Step 2: Predict sentiment
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
    
    # Step 3: Interpret the prediction
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    label_names = sentiment_model.config.id2label
    predicted_label = label_names[predicted_class_id]
    
    # Print the predicted sentiment
    return predicted_label

keyword_tokenizer = AutoTokenizer.from_pretrained("transformer3/H2-keywordextractor")
keyword_model = AutoModelForSeq2SeqLM.from_pretrained("transformer3/H2-keywordextractor")

def extract_keywords(text):
    # Load the tokenizer and model
    
    # Step 1: Tokenize the input text
    inputs = keyword_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Step 2: Generate keywords
    with torch.no_grad():
        outputs = keyword_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    
    # Step 3: Decode the output
    keywords = keyword_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return keywords