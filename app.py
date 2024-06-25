from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
model = load_model('emotion_detection_model.h5')  # Load the saved model

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load encoder
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)


# # Define custom stopwords
# custom_stopwords = set(stopwords.words('english')) - {
#     'not', 'no', 'yes', 'why', 'what', 'how', 'when', 'where', 'which', 'about',
#     'happy', 'sad', 'angry', 'excited', 'extremely', 'very', 'too', 'quite',
#     'devastated', 'thrilled', 'amazing', 'terrible', 'but', 'however', 'although'
# }

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters except spaces
    tokens = word_tokenize(text)  # Tokenize the text
    # tokens = [word for word in tokens if word not in custom_stopwords]  # Remove custom stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the tokens
    return tokens

# Endpoint for prediction
@app.route('/detect', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Preprocess the input text
    preprocessed_input = preprocess_text(text)
    input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=100)
    
    # Make prediction using the loaded model
    predicted_probabilities = model.predict(padded_input_sequence)
    predicted_label = encoder.inverse_transform(predicted_probabilities)[0]

    # Get the confidence score for the predicted emotion
    confidence_score = np.max(predicted_probabilities)
    
    return jsonify({'predicted_emotion': predicted_label[0], 'confidence_score': float(confidence_score)})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
