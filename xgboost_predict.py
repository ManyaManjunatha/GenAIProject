"""
xgboost_predict.py
------------------
Loads the trained XGBoost model + TF-IDF vectorizer from disk
and provides a simple `predict_xgb(text)` function for inference.
"""

import os
import re
import string
import pickle
import numpy as np

# -----------------------------
# 1️⃣ Utility: Clean Text
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# 2️⃣ Load Model + Vectorizer
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "xgb_vectorizer.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        xgb_vectorizer = pickle.load(f)
    print("✅ Loaded XGBoost model and TF-IDF vectorizer.")
except Exception as e:
    print(f"⚠️ Failed to load model/vectorizer: {e}")
    xgb_model = None
    xgb_vectorizer = None


# -----------------------------
# 3️⃣ Predict Function
# -----------------------------
def predict_xgb(text: str):
    """
    Predict misinformation using the trained XGBoost model.
    Returns (label, confidence) as (str, float).
    """
    if xgb_model is None or xgb_vectorizer is None:
        return "Model not loaded", 0.0

    if not text or not text.strip():
        return "Invalid Input", 0.0

    cleaned = clean_text(text)
    X_vec = xgb_vectorizer.transform([cleaned])
    proba = xgb_model.predict_proba(X_vec)[0]
    label_idx = int(np.argmax(proba))
    label = "True News" if label_idx == 1 else "Fake News"
    confidence = float(proba[label_idx])
    return label, confidence


# -----------------------------
# 4️⃣ Local Test (Optional)
# -----------------------------
if __name__ == "__main__":
    sample = "NASA confirms discovery of water on the moon."
    result, conf = predict_xgb(sample)
    print(f"\n📰 Text: {sample}")
    print(f"🔍 Prediction: {result} ({conf*100:.2f}%)")
