# data_preprocessing.py

import pandas as pd
import numpy as np
import re
import nltk
import joblib
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Load and merge all labeled datasets ---
data_files = glob.glob('data/*.csv')
dfs = []

for file in data_files:
    df = pd.read_csv(file, encoding='latin1', low_memory=False)
    if {'subject', 'body', 'label'}.issubset(df.columns):
        df = df[['subject', 'body', 'label']].dropna()
        dfs.append(df)
        print(f"[OK] Loaded {os.path.basename(file)}: {len(df)} rows")
    else:
        print(f"[SKIP] {os.path.basename(file)}: Missing required columns")

# --- Combine all dataframes ---
if not dfs:
    raise ValueError("No valid labeled datasets found.")

df = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO] Combined dataset size: {len(df)} rows")

# --- Clean text ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# --- Combine subject + body and clean ---
print("\n[INFO] Cleaning and preprocessing text...")
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['clean_text'] = df['text'].apply(clean_text)

# --- Remove stopwords ---
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# --- TF-IDF Vectorization ---
print("[INFO] Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].astype(int)

# --- Train/Test Split ---
print("[INFO] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Save outputs ---
print("\n[INFO] Saving processed data and vectorizer...")
joblib.dump((X_train, X_test, y_train, y_test), 'models/train_test_split.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("[DONE] Preprocessing complete. Files saved to models/ directory.")
