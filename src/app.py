import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Sample data to train quickly
spam_data = [
    ("Win money now!!! Free prize", 1),
    ("Congratulations you won lottery", 1),
    ("Urgent! Your account hacked click here", 1),
    ("Free offer click now limited time", 1),
    ("Get rich quick scheme", 1),
    ("Hi, how are you today?", 0),
    ("Let's meet for lunch tomorrow", 0),
    ("Please send me the project report", 0),
    ("Meeting scheduled at 3 PM", 0),
    ("Can you review my code?", 0),
]

texts = [x[0] for x in spam_data]
labels = [x[1] for x in spam_data]

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

st.set_page_config(page_title="Spam Email Detection")
st.title("📧 Spam Email Detection")
st.write("Your Final Project - Spam Email Detection")

email_text = st.text_area("Enter Email Text:", "Congratulations! You won a free iPhone. Click here now!")

if st.button("Check Spam"):
    X_test = vectorizer.transform([email_text])
    pred = model.predict(X_test)[0]
    if pred == 1:
        st.error("🚨 This is SPAM!")
    else:
        st.success("✅ This is NOT Spam - Safe Email")

st.markdown("---")
st.write("Project Title: Spam Email Detection")