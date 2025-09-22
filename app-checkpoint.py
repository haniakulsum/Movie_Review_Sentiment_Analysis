import streamlit as st
import re
import string
import joblib

# Load trained model
model = joblib.load("sentiment_model.pkl")  # Saved pipeline

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="IMDb Review Sentiment Analyzer", page_icon="⭐")
st.title("IMDb Movie Review Sentiment Analysis")
st.write("Enter your movie review below:")

input_text = st.text_area("Review", "")

if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(input_text)
        prediction = model.predict([cleaned])[0]
        if prediction == "positive":
            st.success("This is a Positive Review!")
            st.markdown("### :thumbsup:")
        else:
            st.error("This is a Negative Review!")
            st.markdown("### :thumbsdown:")