
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import html
import hashlib
import logging
import traceback
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Security Configurations ---
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB file size
RATE_LIMIT_SECONDS = 2
MAX_REQUESTS_PER_HOUR = 50

# --- Session Setup ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'request_reset_time' not in st.session_state:
    st.session_state.request_reset_time = datetime.now().timestamp() + 3600

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phishing_detector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Load Model ---
try:
    model = joblib.load('models/mnb_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    X_train, X_test, y_train, y_test = joblib.load('models/train_test_split.pkl')
except Exception as e:
    st.error("Failed to load model or data.")
    raise e

# --- Utility Functions ---
def sanitize_input(text):
    text = html.escape(text)
    return re.sub(r'(&lt;|<)script.*?(&gt;|>)', '', text, flags=re.IGNORECASE)

def check_rate_limit():
    now = datetime.now().timestamp()
    if now > st.session_state.request_reset_time:
        st.session_state.request_count = 0
        st.session_state.request_reset_time = now + 3600
    if (now - st.session_state.last_request_time) < RATE_LIMIT_SECONDS:
        wait_time = RATE_LIMIT_SECONDS - (now - st.session_state.last_request_time)
        return False, f"Please wait {wait_time:.1f} seconds between requests."
    if st.session_state.request_count >= MAX_REQUESTS_PER_HOUR:
        return False, "Request limit reached. Please wait an hour."
    st.session_state.last_request_time = now
    st.session_state.request_count += 1
    return True, ""

def validate_file(uploaded_file):
    if uploaded_file is None:
        return True, ""
    size = len(uploaded_file.getvalue())
    if size > MAX_FILE_SIZE:
        return False, f"File too large. Max size is {MAX_FILE_SIZE // (1024 * 1024)}MB."
    if uploaded_file.type != "text/plain":
        return False, "Only plain .txt files are supported."
    return True, ""

def predict_phishing(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    proba = model.predict_proba(text_vectorized)[0]
    return prediction, proba

def highlight_suspicious_keywords(text):
    feature_names = vectorizer.get_feature_names_out()
    log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    top_phishing_words = pd.Series(log_odds, index=feature_names).sort_values(ascending=False).head(200)
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    matched_keywords = [word for word in top_phishing_words.index if word.lower() in words_in_text]
    highlighted_text = text
    for keyword in matched_keywords:
        highlighted_text = re.sub(
            rf"\b({re.escape(keyword)})\b",
            r'<span style="background-color: #a64654; font-weight: bold;">\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text, matched_keywords

# --- Streamlit UI ---
st.set_page_config(page_title="Phishing Email Detector", layout="wide")
st.title("📧 Phishing Email Detector")
tab1, tab2 = st.tabs(["🔍 Detector", "📚 How it Works"])

# === TAB 1: DETECTOR ===
with tab1:
    st.warning("""
    🔒 **Security Notice:**
    - Do not upload sensitive or confidential information
    - This is a public demo tool - use at your own risk
    - We do not store any submitted emails
    - Rate limits apply to prevent abuse
    """)
    col1, col2 = st.columns(2)
    with col1:
        email_text = st.text_area("Paste email content here:", height=200)
    with col2:
        uploaded_file = st.file_uploader(
            "Or upload a text file:", 
            type=['txt'], 
            help="Only .txt files are supported. Max file size: 5MB."
            , key="file_uploader" )
        if uploaded_file:
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                st.stop()
            email_text = uploaded_file.getvalue().decode()
            st.text_area("File contents:", email_text, height=200)

    show_highlight = st.checkbox("🔍 Highlight suspicious keywords", value=True)

    if st.button("Analyze Email"):
        if not email_text:
            st.error("Please paste or upload email text.")
        else:
            can_proceed, limit_msg = check_rate_limit()
            if not can_proceed:
                st.error(limit_msg)
                st.stop()

            email_text = sanitize_input(email_text)
            prediction, probabilities = predict_phishing(email_text)
            confidence = max(probabilities) * 100
            phishing_prob = probabilities[1] * 100
            label = "Phishing" if prediction == 1 else "Legitimate"

            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'prediction': label,
                'confidence': round(confidence, 1)
            })

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", label)
            m2.metric("Confidence", f"{confidence:.1f}%")
            m3.metric("Phishing Probability", f"{phishing_prob:.1f}%")

            st.markdown("### Analysis")
            highlighted_html, keywords = highlight_suspicious_keywords(email_text)
            st.markdown("**Suspicious keywords found:**")
            st.write(", ".join(keywords))
            st.markdown("### Email Content")
            if show_highlight:
                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                st.code(email_text)

            if len(st.session_state.history) > 1:
                hist_df = pd.DataFrame(st.session_state.history)
                st.markdown("### 🔁 Prediction History (This Session)")
                st.line_chart(hist_df.set_index('timestamp')['confidence'])

# === TAB 2: EXPLANATION ===
with tab2:
    st.header("How the Phishing Detector Works")
    
    # Create two columns for the first row
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Training Data Distribution")
        # Create a smaller pie chart
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        label_counts = pd.Series(y_train.tolist() + y_test.tolist()).value_counts()
        wedges, texts, autotexts = ax1.pie(
            label_counts, 
            labels=["Legitimate", "Phishing"], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=["#2ca02c", "#d62728"]
        )
        # Make the percentage labels easier to read
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=10)
        ax1.axis('equal')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("🤖 Model: Multinomial Naive Bayes")
        st.markdown("""
        Our model uses Multinomial Naive Bayes classification because:
        
        ✨ **Speed & Efficiency**
        - Fast training and prediction
        - Excellent for real-time analysis
        
        📊 **Text Processing**
        - Natural fit for text classification
        - Handles word frequencies effectively
        
        💪 **Reliability**
        - Works well with limited data
        - Robust against irrelevant features
        
        🎯 **Probabilistic Approach**
        - Provides confidence scores
        - Handles uncertainty well
        """)
    
    # Separator
    st.markdown("---")
    
    # Feature importance section with better styling
    st.subheader("🔍 Most Important Phishing Indicators")
    feature_names = vectorizer.get_feature_names_out()
    log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': log_odds})
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    
    # Create a more compact and styled horizontal bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.barh(importance_df['feature'], importance_df['importance'], 
                    color='#a64654', alpha=0.8)
    ax2.set_title('Top Words Indicating Phishing Risk', pad=20)
    ax2.invert_yaxis()  # Highest values at top
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', 
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Separator
    st.markdown("---")
    
    # Model performance section with better organization
    st.subheader("📈 Model Performance Metrics")
    
    # Metrics in columns with explanations
    y_pred = model.predict(X_test)
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        st.caption("Correct predictions out of all predictions")
    
    with m2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        st.caption("Accuracy of phishing predictions")
    
    with m3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        st.caption("Ability to find actual phishing")
    
    with m4:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
        st.caption("Balance of precision & recall")
    
    # Confusion matrix with explanation
    st.markdown("#### Detailed Performance Breakdown")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=["Actually Legitimate", "Actually Phishing"],
            columns=["Predicted Legitimate", "Predicted Phishing"]
        )
        st.dataframe(cm_df.style.background_gradient(cmap='RdYlGn'))
    
    with col4:
        st.markdown("""
        #### Understanding the Results
        
        The confusion matrix shows:
        - **True Negatives**: Correctly identified legitimate emails
        - **False Positives**: Legitimate emails mistakenly flagged
        - **False Negatives**: Missed phishing attempts
        - **True Positives**: Correctly caught phishing
        
        Our model prioritizes:
        - ⚠️ High recall to catch more phishing attempts
        - ✅ Good precision to minimize false alarms
        """)
