# 1. Import libraries
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib

# 2. Load dataset
df = pd.read_csv("IMDB Dataset.csv")  # Ensure this CSV is in the same directory

# 3. Preprocess data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(rf"[{string.punctuation}]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df['review'] = df['review'].apply(clean_text)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 5. Pipeline: Vectorization + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# 6. Train model
model.fit(X_train, y_train)

# 7. Save trained model
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model saved as 'sentiment_model.pkl'")

# 8. Predictions & Evaluation
y_pred = model.predict(X_test)
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📝 Classification Report:\n", classification_report(y_test, y_pred))

# 9. ROC Curve
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test_encoded, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")  # Save ROC curve as image
plt.show()

# 10. Final accuracy
print(f"\n✅ Final Accuracy: {accuracy_score(y_test, y_pred):.4f}")
