### IMDb Sentiment Analysis

A simple end‑to‑end project that trains a sentiment analysis model on the IMDb movie reviews dataset and serves an interactive demo using Streamlit.

The project includes:
- **Training script**: builds a TF‑IDF + Logistic Regression pipeline and saves it as `sentiment_model.pkl`.
- **Evaluation**: prints metrics and saves a ROC curve to `roc_curve.png`.
- **App**: a Streamlit UI in `app.py` to classify custom reviews as Positive/Negative.

---

### Project structure

```
imdb_sentiment_analysis/
  app.py                  # Streamlit app
  train_model.py          # Model training & evaluation
  IMDB Dataset.csv        # Labeled dataset (review, sentiment)
  sentiment_model.pkl     # Saved model pipeline (created by training)
  roc_curve.png           # ROC curve image (created by training)
  README.md               # This file
```

---

### Requirements

Python 3.9+ is recommended. Install dependencies:

```bash
pip install -U pip
pip install streamlit scikit-learn pandas matplotlib joblib
```

If you prefer, create and activate a virtual environment first (Windows PowerShell):

```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

---

### Train the model

1) Ensure `IMDB Dataset.csv` is in the project root (already included).

2) Run training:

```bash
python train_model.py
```

This will:
- Clean reviews and split train/test
- Train TF‑IDF + Logistic Regression
- Print accuracy and a classification report
- Generate `roc_curve.png`
- Save the model pipeline to `sentiment_model.pkl`

> Note: The app expects `sentiment_model.pkl` in the project root.

---

### Run the Streamlit app

After training (or if `sentiment_model.pkl` already exists):

```bash
streamlit run app.py
```

Then open the local URL shown (e.g., `http://localhost:8501`) and paste a movie review to classify.

---

### How it works

- **Preprocessing**: lowercasing, remove HTML tags, punctuation, and extra spaces.
- **Vectorization**: `TfidfVectorizer(max_features=10000, stop_words='english')`.
- **Model**: `LogisticRegression(max_iter=1000)`.
- **Pipeline**: Vectorizer + Classifier in a single `sklearn.pipeline.Pipeline` saved via `joblib`.

---

### Troubleshooting

- **Module not found (streamlit/sklearn/etc.)**: Re‑run the install commands or ensure your venv is activated.
- **Port already in use** when starting Streamlit: run `streamlit run app.py --server.port 8502` (change port as needed).
- **Model not found**: Run the training step to generate `sentiment_model.pkl` in the project root.
- **UnicodeDecodeError loading CSV**: Ensure the CSV is UTF‑8 encoded; try `pd.read_csv('IMDB Dataset.csv', encoding='utf-8')` if needed.

---

### Notes

- The dataset column names expected are `review` and `sentiment` with labels like `positive` and `negative`.
- The Streamlit app applies the same text cleaning as training before prediction.

---

### License

This project is provided as‑is for educational purposes. Add your preferred license if you intend to distribute it.


