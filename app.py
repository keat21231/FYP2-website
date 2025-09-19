import os
import re
import joblib
import requests
import numpy as np
import pandas as pd
import sqlite3
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
from gensim.models import Word2Vec
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# ==================== App init ====================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise ValueError("FLASK_SECRET_KEY environment variable must be set")

# ==================== Config / paths ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "Overall_cleaned.csv")
INSURANCE_PATH = os.path.join(BASE_DIR, "insurance_panel.csv")


LR_PATH           = os.path.join(BASE_DIR, "logistic_model.pkl")
SCALER_PATH       = os.path.join(BASE_DIR, "scaler.pkl")
LABEL_ENCODER_PATH= os.path.join(BASE_DIR, "label_encoder.pkl")
W2V_PATH          = os.path.join(BASE_DIR, "Cleaned_Data_Word2Vec.model")

EXPLAIN_URL = os.getenv("EXPLAIN_URL", "http://localhost:5001/explain")

# ==================== Load data & models ====================
df, insurance_df = None, None
logistic_model, scaler, label_encoder, word2vec_model = None, None, None, None

try:
    df = pd.read_csv(REVIEWS_PATH)
    print(f"[LOAD] Reviews loaded: {len(df)} rows")
except Exception as e:
    print(f"[ERROR] Could not load reviews CSV: {e}")

try:
    insurance_df = pd.read_csv(INSURANCE_PATH)
    print(f"[LOAD] Insurance table loaded: {len(insurance_df)} rows")
except Exception as e:
    print(f"[WARN] Could not load insurance CSV: {e}")

def safe_joblib_load(path, name):
    try:
        obj = joblib.load(path)
        print(f"[LOAD] {name} loaded")
        return obj
    except Exception as e:
        print(f"[WARN] Could not load {name}: {e}")
        return None

logistic_model = safe_joblib_load(LR_PATH, "LR model")
scaler = safe_joblib_load(SCALER_PATH, "Scaler")
label_encoder = safe_joblib_load(LABEL_ENCODER_PATH, "LabelEncoder")

try:
    word2vec_model = Word2Vec.load(W2V_PATH)
    print(f"[LOAD] Word2Vec loaded (vector size {word2vec_model.vector_size})")
except Exception as e:
    print(f"[WARN] Could not load Word2Vec: {e}")

# ==================== SQLite user DB ====================
DB_PATH = os.path.join(BASE_DIR, "users.db")
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ==================== Helpers ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login_route"))  # <-- fix here
        return f(*args, **kwargs)
    return decorated_function


def norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

if df is not None:
    df.columns = [c.strip() for c in df.columns]
    if "Hospital" in df.columns:
        df["hospital_key"] = df["Hospital"].apply(norm_key)
    if "Cleaned Review" not in df.columns and "Cleaned_Review" in df.columns:
        df.rename(columns={"Cleaned_Review": "Cleaned Review"}, inplace=True)

if insurance_df is not None and "Hospital_Name" in insurance_df.columns:
    insurance_df["hospital_key"] = insurance_df["Hospital_Name"].apply(norm_key)

def vectorize_text(text: str) -> np.ndarray:
    if word2vec_model is None:
        return np.zeros(100, dtype=float)
    if text is None:
        return np.zeros(word2vec_model.vector_size, dtype=float)
    tokens = [t.strip().lower() for t in re.split(r"[^\w]+", str(text)) if t.strip()]
    vecs = [word2vec_model.wv[t] for t in tokens if t in word2vec_model.wv.key_to_index]
    if not vecs:
        return np.zeros(word2vec_model.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

# ==================== Explain service ====================
def call_explain_service(review_text: str, timeout=30):
    """Call explain service with increased timeout and truncate long text."""
    safe_text = review_text[:1000]  # limit to avoid LIME delay
    try:
        r = requests.post(EXPLAIN_URL, json={"text": safe_text}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"[XAI] Explain service returned {r.status_code}")
        return None
    except Exception as e:
        print(f"[XAI] Error calling explain service: {e}")
        return None

def get_explanation(review_text: str):
    resp = call_explain_service(review_text)
    if not resp:
        return "No explanation available (service unreachable)."
    if isinstance(resp.get("reason"), str) and resp["reason"].strip():
        return resp["reason"]

    shap_words = resp.get("shap_words", [])
    lime_words = resp.get("lime_words", [])
    parts = []
    if shap_words:
        parts.append(f"Patients often mentioned {', '.join(shap_words[:3])}.")
    if lime_words:
        parts.append(f"LIME also highlighted {', '.join(lime_words[:2])}.")
    return " ".join(parts) if parts else "No explanation available."

# ==================== Core mapping ====================
def map_prediction_to_hospitals(prediction, keywords: str):
    if df is None:
        return []

    review_col = "Cleaned Review" if "Cleaned Review" in df.columns else (
        "Review content" if "Review content" in df.columns else None
    )
    if not review_col:
        return []

    keywords = (keywords or "").strip()
    if not keywords:
        return []
    keyword_list = [k.lower() for k in re.split(r"[,\s]+", keywords) if k]
    if not keyword_list:
        return []

    def token_match(text: str):
        if not isinstance(text, str):
            return False
        tokens = set(text.lower().split())
        return any(k in tokens for k in keyword_list)

    matched = df[df[review_col].apply(token_match)]
    if matched.empty:
        def substr_match(text: str):
            if not isinstance(text, str):
                return False
            low = text.lower()
            return any(k in low for k in keyword_list)
        matched = df[df[review_col].apply(substr_match)]

    if matched.empty:
        return []

    counts = matched.groupby(["Hospital", "hospital_key"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).head(5)

    results = []
    for _, row in counts.iterrows():
        h_name, h_key, h_count = row["Hospital"], row["hospital_key"], int(row["count"])
        hosp_reviews = matched[matched["hospital_key"] == h_key][review_col].astype(str).tolist()

        if hosp_reviews:
            sample_review = max(hosp_reviews, key=len)  # use longest review for explanation
            reason = get_explanation(sample_review)
        else:
            reason = "No explanation available."

        results.append({"hospital": h_name, "count": h_count, "xai": reason})

    return results

# ==================== Routes ====================
# ----------------- LOGIN -----------------
@app.route("/login", methods=["GET", "POST"])
def login_route():
    if request.method == "POST":
        username_email = request.form.get("username_email").strip()
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, username, email, password 
            FROM users 
            WHERE username=? OR email=?
        """, (username_email, username_email))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username/email or password.", "danger")
            return redirect(url_for("login_route"))

    return render_template("login.html")


# ----------------- REGISTER -----------------
@app.route("/register", methods=["GET", "POST"])
def register_route():
    if request.method == "POST":
        username = request.form.get("username").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required!", "warning")
            return redirect(url_for("register_route"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register_route"))

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, generate_password_hash(password)))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login_route"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


# ----------------- LOGOUT -----------------
@app.route("/logout")
def logout_route():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login_route"))


# ----------------- FORGOT PASSWORD -----------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password_route():
    if request.method == "POST":
        email = request.form.get("email").strip()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()

        if user:
            return redirect(url_for("reset_password_route", user_id=user[0]))
        else:
            flash("Email not found.", "danger")
            return redirect(url_for("forgot_password_route"))

    return render_template("forgot-password.html")


# ----------------- RESET PASSWORD -----------------
@app.route("/reset-password/<int:user_id>", methods=["GET", "POST"])
def reset_password_route(user_id):
    if request.method == "POST":
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("reset_password_route", user_id=user_id))

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE id=?", (generate_password_hash(password), user_id))
        conn.commit()
        conn.close()
        flash("Password updated successfully! Please login.", "success")
        return redirect(url_for("login_route"))

    return render_template("reset-password.html", user_id=user_id)


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        keywords = request.form.get("keywords", "").strip()
        if not keywords:
            flash("Please enter keywords to search.", "warning")
            return redirect(url_for("index"))

        try:
            vec = vectorize_text(keywords).reshape(1, -1)
            scaled_for_model = scaler.transform(vec) if scaler is not None else vec
            _pred = logistic_model.predict(scaled_for_model) if logistic_model is not None else None
        except Exception as e:
            print(f"[WARN] prediction failed: {e}")
            _pred = None

        prediction = map_prediction_to_hospitals(_pred, keywords)
        session["index_keywords"] = keywords
        session["index_prediction"] = prediction
        return redirect(url_for("index"))

    keywords = session.pop("index_keywords", None)
    prediction = session.pop("index_prediction", [])
    return render_template("index.html", prediction=prediction, keywords=keywords)

@app.route("/insurance", methods=["GET", "POST"])
@login_required
def insurance():
    if request.method == "POST":
        insurance = request.form.get("insurance", "").strip()
        if not insurance:
            flash("Please enter an insurance provider.", "warning")
            return redirect(url_for("insurance"))

        if insurance_df is None or "Insurance_Company" not in insurance_df.columns:
            hospitals = []
        else:
            hospitals = insurance_df[
                insurance_df["Insurance_Company"].str.contains(insurance, case=False, na=False)
            ]["Hospital_Name"].dropna().drop_duplicates().sort_values().tolist()

        if not hospitals:
            flash(f"No hospitals found for insurance: {insurance}", "warning")

        # ✅ Save current search
        session["ins_name"] = insurance
        session["ins_hospitals"] = hospitals

        # ✅ Track last 3 searches
        recent = session.get("recent_providers", [])
        if insurance not in recent:   # avoid duplicates
            recent.insert(0, insurance)   # add to front
        recent = recent[:3]  # keep max 3
        session["recent_providers"] = recent

        return redirect(url_for("insurance"))

    insurance = session.pop("ins_name", None)
    hospitals = session.pop("ins_hospitals", [])
    recent_providers = session.get("recent_providers", [])

    # Get provider list dynamically (if available)
    providers = []
    if insurance_df is not None and "Insurance_Company" in insurance_df.columns:
        providers = insurance_df["Insurance_Company"].dropna().drop_duplicates().sort_values().tolist()

    return render_template("insurance.html",
                            insurance=insurance,
                            hospitals=hospitals,
                            recent_providers=recent_providers,
                            providers=providers)


@app.route("/insurance/clear", methods=["POST"])
@login_required
def clear_recent():
    session.pop("recent_providers", None)
    flash("Recent insurance history cleared!", "info")
    return redirect(url_for("insurance"))


@app.route("/hospital-info")
@login_required
def hospital_info():
    return render_template("hospital-info.html")

@app.route("/api/recommend", methods=["POST"])
@login_required
def api_recommend():
    data = request.get_json() or {}
    keywords = data.get("keywords", "")
    vec = vectorize_text(keywords).reshape(1, -1)
    try:
        scaled_for_model = scaler.transform(vec) if scaler is not None else vec
        _ = logistic_model.predict(scaled_for_model) if logistic_model is not None else None
    except Exception:
        _ = None
    results = map_prediction_to_hospitals(_ if _ is not None else None, keywords)
    return jsonify(results)


# ==================== Run ====================
if __name__ == "__main__":
    app.run(debug=True)
