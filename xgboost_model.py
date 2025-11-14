"""
xgboost_model.py
----------------
Misinformation Detection Model using XGBoost + TF-IDF
Improved for stability, cleaner preprocessing, and balanced training.
"""

import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# -----------------------------
# 1️⃣ Paths & Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WELFake_Dataset.csv")

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "title"]).reset_index(drop=True)
df["clean_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
 
print(f"✅ Loaded {len(df)} total articles.")

# -----------------------------
# 2️⃣ Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """Cleans and normalizes input text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove digits
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

print("🧹 Cleaning text data...")
tqdm.pandas()
df["clean_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).progress_apply(clean_text)

# -----------------------------
# 3️⃣ Train/Test Split
# -----------------------------
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -----------------------------
# 4️⃣ TF-IDF Vectorization
# -----------------------------
print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5️⃣ Handle Class Imbalance
# -----------------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]
print(f"⚖️  Class Weights: {class_weight_dict}")

# -----------------------------
# 6️⃣ Train XGBoost Model
# -----------------------------
print("🚀 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

# -----------------------------
# 7️⃣ Evaluation
# -----------------------------
print("📈 Evaluating model...")
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Fake", "True"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 8️⃣ Save Model and Vectorizer
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "xgb_vectorizer.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(VEC_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n💾 Saved model → {MODEL_PATH}")
print(f"💾 Saved vectorizer → {VEC_PATH}")
print("\n✅ Training complete. Model ready for inference.")
