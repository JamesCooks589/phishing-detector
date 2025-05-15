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
        print(f"✅ Loaded {os.path.basename(file)}: {len(df)} rows")
    else:
        print(f"❌ Skipped {os.path.basename(file)}: Missing required columns")

# --- Combine all dataframes ---
if not dfs:
    raise ValueError("No valid labeled datasets found.")
df = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Combined dataset size: {len(df)} emails")

# --- Combine subject + body ---
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# --- Clean text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetical characters
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

print("\n✅ Preprocessing complete. Combined data and vectorizer saved.")
