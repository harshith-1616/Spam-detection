"""
Spam Detection — Streamlit app
Run with:  streamlit run app.py
"""

import re
import string
import sys

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "Spam_detection.joblib"


# ---------------------------------------------------------------------------
# IMPORTANT: the saved pipeline's CountVectorizer holds a reference to a
# function named `wordopt` that lives in __main__. It is NOT stored inside the
# .joblib file, so it must exist here BEFORE joblib.load() is called, otherwise
# loading fails with:
#   AttributeError: Can't get attribute 'wordopt' on <module '__main__'>
#
# This is the exact wordopt() used during training — keep it in sync with the
# training script, since the vectorizer's vocabulary was built with it.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Streamlit reruns the script on every interaction and the module name is not
# always __main__, so register the function explicitly to be safe.
sys.modules["__main__"].wordopt = wordopt


@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)


def label_of(pred):
    return "SPAM" if int(pred) == 1 else "HAM"


st.set_page_config(page_title="Spam Detection", page_icon="📩")
st.title("📩 Spam Detection")
st.caption("CountVectorizer + RandomForest pipeline")

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Could not find `{MODEL_PATH}`. Put it next to app.py.")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load the model: {e}")
    st.stop()

tab_single, tab_batch = st.tabs(["Single message", "Batch (CSV)"])

with tab_single:
    message = st.text_area(
        "Message",
        height=160,
        placeholder="Paste an SMS or email here...",
    )

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.warning("Type a message first.")
        else:
            pred = model.predict([message])[0]
            proba = model.predict_proba([message])[0]
            spam_prob = float(proba[list(model.classes_).index(1)])

            if int(pred) == 1:
                st.error(f"🚨 SPAM — {spam_prob:.0%} confidence")
            else:
                st.success(f"✅ HAM (not spam) — {1 - spam_prob:.0%} confidence")

            st.progress(spam_prob, text=f"Spam probability: {spam_prob:.1%}")

            with st.expander("Cleaned text the model actually sees"):
                st.code(wordopt(message) or "(empty after cleaning)")

with tab_batch:
    st.write("Upload a CSV and pick the column holding the message text.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

        column = st.selectbox("Text column", df.columns)

        if st.button("Classify all"):
            texts = df[column].fillna("").astype(str)
            preds = model.predict(texts)
            probas = model.predict_proba(texts)
            spam_idx = list(model.classes_).index(1)

            out = df.copy()
            out["prediction"] = [label_of(p) for p in preds]
            out["spam_probability"] = probas[:, spam_idx].round(3)

            st.dataframe(out, use_container_width=True)
            st.info(f"{(out['prediction'] == 'SPAM').sum()} of {len(out)} flagged as spam.")

            st.download_button(
                "Download results",
                out.to_csv(index=False).encode("utf-8"),
                file_name="spam_predictions.csv",
                mime="text/csv",
            )
