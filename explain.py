from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import random
import pandas as pd
from gensim.models import Word2Vec
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# ==================== Config ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LR_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
W2V_PATH = os.path.join(BASE_DIR, "Cleaned_Data_Word2Vec.model")
DATASET_PATH = os.path.join(BASE_DIR, "Overall_cleaned.csv")

# ==================== Load models ====================
def safe_load(path, name):
    try:
        obj = joblib.load(path)
        print(f"[LOAD] {name} loaded from {path}")
        return obj
    except Exception as e:
        print(f"[WARN] Could not load {name}: {e}")
        return None

logistic_model = safe_load(LR_PATH, "LR model")
scaler = safe_load(SCALER_PATH, "Scaler")

try:
    w2v = Word2Vec.load(W2V_PATH)
    print(f"[LOAD] Word2Vec loaded (vector size {w2v.vector_size})")
except Exception as e:
    print(f"[ERROR] Could not load Word2Vec: {e}")
    w2v = None

try:
    train_df = pd.read_csv(DATASET_PATH)
    print(f"[LOAD] Dataset loaded: {len(train_df)} rows")
except Exception as e:
    print(f"[WARN] Could not load dataset: {e}")
    train_df = pd.DataFrame()

# ==================== Helpers ====================
def preprocess_text(text: str) -> str:
    """Limit review length to reduce LIME overhead."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:100])  # keep only first 100 words

def text_to_vector(text: str) -> np.ndarray:
    """Convert text into average Word2Vec embedding."""
    if not isinstance(text, str) or not text.strip() or w2v is None:
        return np.zeros(100)
    words = text.lower().split()
    vectors = [w2v.wv[w] for w in words if w in w2v.wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

def shap_word_importance(text: str):
    """Approximate SHAP importance for linear model."""
    words = text.lower().split()
    importance = {}
    for w in words:
        if w in w2v.wv.key_to_index:
            vec = w2v.wv[w].reshape(1, -1)
            scaled = scaler.transform(vec)
            contribution = float(np.dot(scaled, logistic_model.coef_.T).flatten()[0])
            importance[w] = abs(contribution)
    return sorted(importance, key=importance.get, reverse=True)

def make_human_reason(shap_words, lime_words):
    """Create simple explanation for non-technical users."""
    top_keywords = list(dict.fromkeys(shap_words[:3] + lime_words[:2]))
    if not top_keywords:
        return "Patients generally reported positive experiences with this hospital."
    return (
        f"Patients often mentioned {', '.join(top_keywords)} as important factors "
        f"in their overall hospital experience."
    )

# ==================== Routes ====================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Explain server is running. Use POST /explain"})

@app.route("/explain", methods=["POST"])
def explain_prediction():
    try:
        if logistic_model is None or w2v is None or scaler is None:
            return jsonify({"error": "Model, scaler, or Word2Vec not loaded"}), 500

        data = request.get_json() or {}

        # === Case 1: Direct review explanation ===
        if "text" in data:
            text = preprocess_text(data.get("text", "").strip())
            print(f"[DEBUG] Received text (truncated): {text[:150]}...")

            shap_words = shap_word_importance(text)

            lime_words = []
            try:
                class_names = ["Negative", "Positive"]

                def predict_fn(texts):
                    vecs = [text_to_vector(t) for t in texts]
                    scaled = scaler.transform(vecs)
                    return logistic_model.predict_proba(scaled)

                explainer_lime = LimeTextExplainer(class_names=class_names)
                exp = explainer_lime.explain_instance(
                    text,
                    classifier_fn=predict_fn,
                    num_features=5,
                    num_samples=500
                )
                lime_words = [w for w, _ in exp.as_list()]
            except Exception as e:
                print(f"[WARN] LIME explanation failed: {e}")

            reason = make_human_reason(shap_words, lime_words)
            keywords = list(dict.fromkeys(shap_words + lime_words))

            return jsonify({
                "reason": reason,
                "keywords": keywords
            })

        # === Case 2: Keyword search (top hospitals) ===
        elif "keyword" in data:
            keyword = data["keyword"].lower()
            print(f"[DEBUG] Searching hospitals for keyword: {keyword}")

            if "Hospital" not in train_df.columns or "Cleaned Review" not in train_df.columns:
                return jsonify({"error": "Dataset must contain 'Hospital' and 'Cleaned Review' columns"}), 500

            matches = train_df[train_df["Cleaned Review"].str.contains(keyword, case=False, na=False)]
            if matches.empty:
                return jsonify({"keyword": keyword, "results": []})

            grouped = matches.groupby("Hospital")
            results = []

            for hospital, group in grouped:
                review = random.choice(group["Cleaned Review"].tolist())
                text = preprocess_text(review)

                shap_words = shap_word_importance(text)

                lime_words = []
                try:
                    class_names = ["Negative", "Positive"]

                    def predict_fn(texts):
                        vecs = [text_to_vector(t) for t in texts]
                        scaled = scaler.transform(vecs)
                        return logistic_model.predict_proba(scaled)

                    explainer_lime = LimeTextExplainer(class_names=class_names)
                    exp = explainer_lime.explain_instance(
                        text,
                        classifier_fn=predict_fn,
                        num_features=3,
                        num_samples=300
                    )
                    lime_words = [w for w, _ in exp.as_list()]
                except Exception as e:
                    print(f"[WARN] LIME explanation failed for hospital {hospital}: {e}")

                reason = make_human_reason(shap_words, lime_words)
                keywords = list(dict.fromkeys(shap_words + lime_words))

                results.append({
                    "hospital": hospital,
                    "review_used": review,
                    "keywords": keywords,
                    "reason": reason,
                    "mentions": len(group)
                })

            # sort by mentions and take top 5
            results = sorted(results, key=lambda x: x["mentions"], reverse=True)[:5]

            return jsonify({"keyword": keyword, "results": results})

        else:
            return jsonify({"error": "Provide either 'text' or 'keyword'"}), 400

    except Exception as e:
        print(f"[ERROR][/explain] {e}")
        return jsonify({"error": str(e)}), 500

# ==================== Run ====================
if __name__ == "__main__":
    app.run(port=5001, debug=True)
