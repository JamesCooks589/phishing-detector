import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the model and vectorizer
model = joblib.load('models/mnb_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_phishing(text):
    # Clean and vectorize the text
    text_vectorized = vectorizer.transform([text])
    # Get prediction and probability
    prediction = model.predict(text_vectorized)[0]
    proba = model.predict_proba(text_vectorized)[0]
    return prediction, proba

def highlight_suspicious_keywords(text):
    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    # Get the most important features for phishing based on the model coefficients
    important_features = pd.DataFrame(
        model.feature_log_prob_[1] - model.feature_log_prob_[0],
        index=feature_names
    ).sort_values(0, ascending=False)
    
    # Get top suspicious keywords
    suspicious_keywords = important_features.head(20).index.tolist()
    
    # Highlight keywords in text
    highlighted_text = text
    for keyword in suspicious_keywords:
        if keyword in text.lower():
            highlighted_text = highlighted_text.replace(
                keyword,
                f'**{keyword}**'  # Markdown bold syntax
            )
    return highlighted_text, suspicious_keywords

# Set up the Streamlit page
st.set_page_config(page_title="Phishing Email Detector", layout="wide")
st.title("📧 Phishing Email Detector")
st.write("Paste an email or upload a text file to check if it's a phishing attempt.")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Text input option
    email_text = st.text_area("Paste email content here:", height=200)
    
with col2:
    # File upload option
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
    if uploaded_file:
        email_text = uploaded_file.getvalue().decode()
        st.text_area("File contents:", email_text, height=200)

# Add an analyze button
if st.button("Analyze Email"):
    if email_text:
        # Get prediction
        prediction, probabilities = predict_phishing(email_text)
        
        # Create three columns for metrics
        m1, m2, m3 = st.columns(3)
        
        # Display prediction
        with m1:
            st.metric(
                "Prediction",
                "Phishing" if prediction == 1 else "Legitimate",
                delta="High Risk" if prediction == 1 else "Low Risk"
            )
        
        # Display confidence
        with m2:
            confidence = max(probabilities) * 100
            st.metric(
                "Confidence",
                f"{confidence:.1f}%",
                delta=f"{'↑' if confidence > 75 else '↓'} {confidence:.1f}%"
            )
        
        # Display phishing probability
        with m3:
            phishing_prob = probabilities[1] * 100
            st.metric(
                "Phishing Probability",
                f"{phishing_prob:.1f}%",
                delta=f"{'↑' if phishing_prob > 50 else '↓'} {phishing_prob:.1f}%"
            )
        
        # Highlight suspicious keywords
        highlighted_text, keywords = highlight_suspicious_keywords(email_text)
        
        # Display analysis
        st.markdown("### Analysis")
        st.markdown("**Suspicious keywords found:**")
        st.write(", ".join(keywords))
        
        st.markdown("### Email Content with Highlighted Keywords")
        st.markdown(highlighted_text)
    else:
        st.error("Please enter some email content or upload a file first.")