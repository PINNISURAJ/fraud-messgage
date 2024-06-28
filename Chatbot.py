import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load the dataset
df = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

# Map labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Function to clean text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['message'] = df['message'].apply(preprocess_text)

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Function to predict new input
def predict_message(message):
    processed_message = preprocess_text(message)
    message_tfidf = tfidf.transform([processed_message])
    prediction = model.predict(message_tfidf)[0]
    return "spam" if prediction == 1 else "ham"

# Streamlit interface
st.title("SMS Spam Detection")

# Input text box
input_message = st.text_area("Enter a message:")

# Initialize history list
history = []

if st.button("Predict"):
    if input_message:
        prediction = predict_message(input_message)
        history.append((input_message, prediction))
        if prediction == "spam":
            st.markdown(
                f'<div style="background-color: red; padding: 10px; border-radius: 5px;">The message is: {prediction}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: green; padding: 10px; border-radius: 5px;">The message is: {prediction}</div>',
                unsafe_allow_html=True
            )
    else:
        st.write("Please enter a message to predict.")

# History page
if st.button("View History"):
    st.title("Prediction History")
    for msg, pred in history:
        if pred == "spam":
            st.markdown(
                f'<div style="background-color: red; padding: 10px; border-radius: 5px;">Message: {msg}<br>Prediction: {pred}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: green; padding: 10px; border-radius: 5px;">Message: {msg}<br>Prediction: {pred}</div>',
                unsafe_allow_html=True
            )
