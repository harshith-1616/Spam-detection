# 📩 Spam Detection

A simple Streamlit web app that classifies SMS/email messages as **spam** or **ham** (not spam).

🔗 **Live demo:** https://spam-detection-m7dorwkrrwb45yh2en6h72.streamlit.app/

## Features

- Classify a single message and see the spam probability
- Batch-classify a CSV file and download the results
- View the cleaned text the model actually sees

## Model

A scikit-learn `Pipeline`:

| Step | Component |
|---|---|
| Vectorizer | `CountVectorizer` with a custom `wordopt` preprocessor |
| Classifier | `RandomForestClassifier` (`class_weight='balanced'`) |

Labels: `1` = spam, `0` = ham.

The `wordopt` preprocessor lowercases text and strips URLs, HTML tags, bracketed text, punctuation, and words containing digits.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

Make sure `Spam_detection.joblib` is in the same folder as `app.py`.

## Files

```
app.py                   # Streamlit app
Spam_detection.joblib    # Trained pipeline
requirements.txt         # Dependencies
```

## Note

The `wordopt` function must be defined in `app.py` before the model is loaded — the pipeline references it but does not store it.
