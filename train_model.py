# train_model.py

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Load the data ---
X_train, X_test, y_train, y_test = joblib.load('models/train_test_split.pkl')

# --- Train the model ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Model trained.")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- Display confusion matrix ---
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save the model ---
joblib.dump(model, 'models/mnb_model.pkl')
print("\n✅ Model saved as models/mnb_model.pkl")
