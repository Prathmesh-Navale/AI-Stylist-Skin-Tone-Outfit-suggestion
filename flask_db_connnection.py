
from flask import Flask, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import bcrypt

# Local modules
from detect_skin_webcam import detect_skin_tone
from recommend_outfit_images import recommend_images

# ========================
# App Config
# ========================
app = Flask(__name__, static_folder="static", template_folder="templates")

# ⚠️ Use a strong secret key. You can also set it via ENV: FLASK_SECRET
app.secret_key = os.getenv("FLASK_SECRET", "change_me_to_a_long_random_secret")

# ✅ MySQL connection via SQLAlchemy (PyMySQL driver)
# Example env override: export DATABASE_URL='mysql+pymysql://user:pass@127.0.0.1:3307/dbname'
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:Kallu%40111@127.0.0.1:3307/k310",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize DB
db = SQLAlchemy(app)


# ========================
# Models
# ========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __init__(self, email: str, password: str):
        self.email = email.strip().lower()
        # Store hashed password
        self.password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def check_password(self, password: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), self.password.encode("utf-8"))
        except Exception:
            return False


# Create tables if not exist
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        # Print server-side for debugging; avoid leaking details to users
        print("DB init error:", e)


# ========================
# Helpers (ML glue)
# ========================

def process_skin_tone_and_recommend():
    """Capture/predict skin tone and return structured outfit suggestions."""
    skin_tone = detect_skin_tone()  # from detect_skin_webcam.py

    outfit_dict = recommend_images(skin_tone)  # from recommend_outfit_images.py

    # Ensure paths are browser-usable (prefix with '/'). If paths already start with 'static/',
    # '/static/..' is preferable for JSON consumers.
    def web_path(p: str) -> str:
        return p if p.startswith("/") else "/" + p.replace("\\", "/")

    cleaned = {"male": {}, "female": {}}
    for gender in outfit_dict:
        for category, imgs in outfit_dict[gender].items():
            cleaned.setdefault(gender, {})[category] = [web_path(img) for img in imgs]

    return {
        "skin_tone": skin_tone,
        "male_outfits": cleaned.get("male", {}),
        "female_outfits": cleaned.get("female", {}),
    }


# ========================
# Routes
# ========================
@app.route("/")
def index():
    # templates/index.html
    return render_template("index.html")



@app.route('/virtual')
def tryon():
    return render_template('virtual.html') 

@app.route("/signup", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template("signup.html", error="Email and password required")

        if User.query.filter_by(email=email).first():
            return render_template("signup.html", error="Email already registered")

        try:
            new_user = User(email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect("/login")
        except Exception as e:
            db.session.rollback()
            print("Signup error:", e)
            return render_template("signup.html", error="Could not sign up. Try a different email.")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session["email"] = user.email
            session["user_id"] = user.id
            return redirect("/front_end")
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


@app.route("/front_end")
def dashboard():
    if "email" in session:
        user = User.query.filter_by(email=session["email"]).first()
        # templates/front_end.html should exist in your project
        return render_template("front_end.html", user=user)
    return redirect("/login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route('/detect', methods=['POST'])
def detect():
    # Step 1: Detect skin tone from webcam
    skin_tone = detect_skin_tone()

    # Step 2: Get recommended outfits for skin tone
    outfit_dict = recommend_images(skin_tone)

    # Step 3: Clean paths (remove "static/" so frontend can prepend it)
    for gender in outfit_dict:
        for category in outfit_dict[gender]:
            outfit_dict[gender][category] = [
                img.replace('static/', '') for img in outfit_dict[gender][category]
            ]

    # Step 4: Send response
    return jsonify({
        'skin_tone': skin_tone,
        'male_outfits': outfit_dict['male'],
        'female_outfits': outfit_dict['female']
    })

from openai import OpenAI
import os

client = OpenAI(api_key=(""))  # store in ENV for securit

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("query", "")
    language = data.get("lang", "en")  # default to English
    skin_tone = session.get("skin_tone", "brown")

    prompt = f"You are a fashion stylist AI. User skin tone: {skin_tone}. Language: {language}. Question: {user_input}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful fashion stylist."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    return jsonify({"answer": answer})



if __name__ == "__main__":
    # Use host/port as needed; debug=True for development only
    app.run(debug=True)
