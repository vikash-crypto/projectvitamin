import os

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import SeparableConv2D
from openai import OpenAI
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import secrets

# ─── OpenAI ───────────────────────────────────────────────────────────────────


# ─── Custom Keras Layer Fix ───────────────────────────────────────────────────
class FixedSeparableConv2D(SeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)

# ─── Flask App ────────────────────────────────────────────────────────────────
import os as _os
_os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'   # allow OAuth over HTTP locally

app = Flask(__name__)
app.secret_key = 'vitamin-app-secret-key-2024-fixed'  # fixed key so session survives redirects

# Session cookie settings — required for OAuth CSRF state to work locally
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE']   = False   # False for local HTTP

# ─── Database ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'vitamin.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)

# ─── Models ───────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    google_id     = db.Column(db.String(120), unique=True, nullable=False)
    name          = db.Column(db.String(200))
    email         = db.Column(db.String(200))
    avatar        = db.Column(db.String(500))
    predictions   = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    predicted_class  = db.Column(db.String(200))
    confidence_score = db.Column(db.Float)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ─── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = 'index'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Google OAuth (Authlib) ─────────────────────────────────────────────────

app.config['GOOGLE_CLIENT_ID']     = GOOGLE_CLIENT_ID
app.config['GOOGLE_CLIENT_SECRET'] = GOOGLE_CLIENT_SECRET

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# ─── Load Model ────────────────────────────────────────────────────────────────
model = load_model(
    os.path.join(BASE_DIR, 'hypervision_vitamin_model.h5'),
    custom_objects={'SeparableConv2D': FixedSeparableConv2D}
)

img_size   = (299, 299)
class_dict = {
    0: 'Normal Nail',
    1: 'Normal Skin',
    2: 'Vitamin B12 Deficiency Skin',
    3: 'Vitamin C Deficiency Nail'
}
classes = list(class_dict.values())

# ─── Multilingual Info ────────────────────────────────────────────────────────
MULTILINGUAL_INFO = {
    "en": {
        "Vitamin B12 Deficiency Skin": {
            "causes":    "Poor dietary intake (vegetarian or vegan diet).",
            "symptoms":  "Fatigue, numbness, pale/yellow skin, mouth ulcers.",
            "treatment": "Increase intake of eggs, fish, poultry, fortified foods, or supplements."
        },
        "Vitamin C Deficiency Nail": {
            "causes":    "Low fruit/vegetable intake, smoking.",
            "symptoms":  "Brittle nails, bleeding gums, slow wound healing.",
            "treatment": "Eat citrus fruits, berries, bell peppers, or take supplements."
        },
        "Normal Skin": {
            "causes":    "No visible deficiency.",
            "symptoms":  "Skin looks healthy.",
            "treatment": "Maintain balanced diet and hydration."
        },
        "Normal Nail": {
            "causes":    "No visible deficiency.",
            "symptoms":  "Nails look healthy.",
            "treatment": "Continue with a nutrient-rich diet."
        }
    },
    "ta": {
        "Vitamin B12 Deficiency Skin": {
            "causes":    "சாகசி அல்லது சைவ உணவு காரணமாக போதிய B12 இல்லை.",
            "symptoms":  "சோர்வு, மரமுட்டல், வெளிர் தோல், வாய் புண்கள்.",
            "treatment": "முட்டை, மீன், கோழி, பால் பொருட்கள் அல்லது B12 சத்து மாத்திரைகள் எடுக்கவும்."
        },
        "Vitamin C Deficiency Nail": {
            "causes":    "பழம் மற்றும் காய்கறிகள் குறைவாக சாப்பிடுதல், புகைபிடித்தல்.",
            "symptoms":  "உடைகிற நகங்கள், ஈறுகளில் இரத்தம், மெதுவான காயம் ஆறுதல்.",
            "treatment": "ஆரஞ்சு, கொய்யா, மிளகாய், ப்ரோக்கோலி சாப்பிடுங்கள்."
        },
        "Normal Skin": {
            "causes":    "குறைபாடு இல்லை.",
            "symptoms":  "தோல் ஆரோக்கியமாக உள்ளது.",
            "treatment": "சீரான உணவும் தண்ணீரும் அருந்துங்கள்."
        },
        "Normal Nail": {
            "causes":    "குறைபாடு இல்லை.",
            "symptoms":  "நகங்கள் ஆரோக்கியமாக உள்ளன.",
            "treatment": "சத்தான உணவை தொடருங்கள்."
        }
    },
    "hi": {
        "Vitamin B12 Deficiency Skin": {
            "causes":    "शाकाहारी या वीगन भोजन से B12 की कमी।",
            "symptoms":  "थकान, सुन्नपन, पीली/पीली त्वचा, मुंह के छाले।",
            "treatment": "अंडे, मछली, मुर्गी, दूध उत्पाद या B12 सप्लीमेंट लें।"
        },
        "Vitamin C Deficiency Nail": {
            "causes":    "फल/सब्जियां कम खाना, धूम्रपान।",
            "symptoms":  "कमजोर नाखून, मसूढ़ों से खून, धीमी घाव भरना।",
            "treatment": "संतरा, अमरूद, शिमला मिर्च, ब्रोकली खाएं।"
        },
        "Normal Skin": {
            "causes":    "कोई कमी नहीं।",
            "symptoms":  "त्वचा स्वस्थ दिखती है।",
            "treatment": "संतुलित आहार और पानी पीते रहें।"
        },
        "Normal Nail": {
            "causes":    "कोई कमी नहीं।",
            "symptoms":  "नाखून स्वस्थ हैं।",
            "treatment": "पोषण युक्त आहार जारी रखें।"
        }
    }
}

# ═══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════

REDIRECT_URI = 'http://localhost:5000/auth/google/callback'

@app.route('/auth/google')
def auth_google():
    return google.authorize_redirect(REDIRECT_URI)


@app.route('/auth/google/callback')
def auth_google_callback():
    try:
        token = google.authorize_access_token()
    except Exception:
        # Stale session / state mismatch — restart the login flow
        return redirect(url_for('auth_google'))
    user_info = token.get('userinfo')
    if not user_info:
        return redirect('/')

    google_id = user_info.get('sub')
    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User(
            google_id=google_id,
            name=user_info.get('name', ''),
            email=user_info.get('email', ''),
            avatar=user_info.get('picture', '')
        )
        db.session.add(user)
        db.session.commit()

    login_user(user, remember=True)
    return redirect('/')


@app.route('/logout')
def logout():
    logout_user()
    return redirect('/')


@app.route('/me')
def me():
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'name': current_user.name,
            'email': current_user.email,
            'avatar': current_user.avatar
        })
    return jsonify({'logged_in': False})


# ═══════════════════════════════════════════════════════════
#  HISTORY ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/history')
def history():
    if not current_user.is_authenticated:
        return jsonify([])
    records = (PredictionHistory.query
               .filter_by(user_id=current_user.id)
               .order_by(PredictionHistory.timestamp.desc())
               .limit(50)
               .all())
    return jsonify([
        {
            'id': r.id,
            'predicted_class': r.predicted_class,
            'confidence_score': r.confidence_score,
            'timestamp': r.timestamp.strftime('%d %b %Y, %I:%M %p')
        }
        for r in records
    ])


@app.route('/history/clear', methods=['POST'])
def history_clear():
    if not current_user.is_authenticated:
        return jsonify({'ok': False, 'error': 'Not logged in'})
    PredictionHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'ok': True})


# ═══════════════════════════════════════════════════════════
#  MAIN ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    if 'image' not in file.content_type:
        return jsonify({'error': 'Uploaded file is not an image'})

    try:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction      = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = classes[predicted_class]
        confidence      = round(float(np.max(prediction) * 100), 2)

        lang = request.form.get('lang', 'en')
        if lang not in MULTILINGUAL_INFO:
            lang = 'en'
        info = MULTILINGUAL_INFO[lang]

        # ── Save to history if user is logged in ──────────────────────
        if current_user.is_authenticated:
            record = PredictionHistory(
                user_id=current_user.id,
                predicted_class=predicted_label,
                confidence_score=confidence
            )
            db.session.add(record)
            db.session.commit()

        return jsonify({
            'predicted_class':  predicted_label,
            'confidence_score': confidence,
            'details':          info.get(predicted_label, {})
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})


# ────────────────────── CHATBOT ──────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    lang         = request.json.get("lang", "en")

    lang_instruction = {
        "en": "Reply in English.",
        "ta": "Reply in Tamil (தமிழில் பதிலளிக்கவும்).",
        "hi": "Reply in Hindi (हिंदी में जवाब दें)."
    }.get(lang, "Reply in English.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful health assistant. Provide safe general information only. Do NOT diagnose or replace professional medical advice. {lang_instruction}"},
                {"role": "user",   "content": user_message}
            ]
        )
        return jsonify({"reply": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Error contacting the AI: {str(e)}"})


# ────────────────────── DETAILS (lang switch) ────────────
@app.route('/details', methods=['GET'])
def details():
    class_name = request.args.get('class_name', '')
    lang       = request.args.get('lang', 'en')
    if lang not in MULTILINGUAL_INFO:
        lang = 'en'
    return jsonify(MULTILINGUAL_INFO[lang].get(class_name, {}))


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
