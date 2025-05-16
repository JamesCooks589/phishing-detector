# explain_model.py

import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('models/mnb_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get log probabilities of features per class
# Index 1 = phishing class (assuming label 1 is phishing)
log_probs = model.feature_log_prob_[1]

# Top suspicious words (sorted by highest log-prob)
def get_top_phishing_words(n=20):
    top_indices = np.argsort(log_probs)[-n:][::-1]
    return [(feature_names[i], round(log_probs[i], 4)) for i in top_indices]

# Test run
if __name__ == "__main__":
    print("[INFO] Top words most indicative of phishing emails:\n")
    for word, weight in get_top_phishing_words(20):
        print(f"{word}: {weight}")
