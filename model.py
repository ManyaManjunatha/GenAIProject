import wikipedia
import numpy as np
import pandas as pd
import re
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class MisinformationDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, true_path='True.csv', fake_path='Fake.csv'):
        """Load and combine real/fake datasets"""
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df['label'] = 'real'
        fake_df['label'] = 'fake'
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.dropna(subset=['text'])
        print(f" Loaded {len(df)} articles")
        return df

    def train_model(self, df):
        """Train a Random Forest on TF-IDF features"""
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'],
                                                            test_size=0.2, random_state=42)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        print("\n Classification Report:\n")
        print(classification_report(y_test, y_pred))

        # Save model and vectorizer
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(" Model and vectorizer saved successfully!")

    # ---------------- Wikipedia + Verification ----------------
    def clean_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    def check_with_wikipedia(self, statement):
        try:
            search_results = wikipedia.search(statement)
            if not search_results:
                return None, None, None
            title = search_results[0]
            summary = wikipedia.summary(title, sentences=2)
            link = wikipedia.page(title).url
            return title, summary, link
        except Exception:
            return None, None, None

    def verify(self, statement):
        """Run misinformation detection + semantic verification"""
        if self.model is None or self.vectorizer is None:
            with open("model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)

        X_vec = self.vectorizer.transform([statement])
        model_pred = self.model.predict(X_vec)[0]
        model_prob = self.model.predict_proba(X_vec)[0]
        confidence = max(model_prob)

        title, summary, link = self.check_with_wikipedia(statement)
        if summary:
            emb1 = self.semantic_model.encode(self.clean_text(statement), convert_to_tensor=True)
            emb2 = self.semantic_model.encode(self.clean_text(summary), convert_to_tensor=True)
            similarity = float(util.cos_sim(emb1, emb2).item())
        else:
            similarity = 0.0

        final_score = 0.6 * confidence + 0.4 * similarity
        label = "Misinformation" if final_score < 0.5 else "Likely True"

        return {
            "input": statement,
            "model_label": model_pred,
            "model_confidence": round(confidence * 100, 2),
            "wikipedia_article": title,
            "similarity_score": round(similarity, 2),
            "final_label": label,
            "final_score": round(final_score * 100, 2),
            "wiki_summary": summary,
            "wiki_link": link
        }

# ---------------- Test ----------------
if __name__ == "__main__":
    detector = MisinformationDetector()
    df = detector.load_data("True.csv", "Fake.csv")
    detector.train_model(df)

    user_input = "Breaking news Mango is a fruit"
    result = detector.verify(user_input)

    print("\n Analysis Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
