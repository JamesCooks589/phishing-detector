# data_preprocessing.py

import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Load and Inspect Data ---
df = pd.read_csv('data/CEAS_08.csv', encoding='latin1')  # Adjust encoding if needed
df = df[['subject', 'body', 'label']]  # Keep relevant columns
df.dropna(inplace=True)

# --- Combine subject + body into one field ---
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# --- Clean text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# --- Remove stopwords ---
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].astype(int)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Save outputs ---
joblib.dump((X_train, X_test, y_train, y_test), 'models/train_test_split.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("✅ Preprocessing complete. Data and vectorizer saved.")
