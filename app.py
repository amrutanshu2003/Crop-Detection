from flask import Flask, render_template, request, redirect, url_for, session, send_file
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import requests
import io

app = Flask(__name__)
app.secret_key = "secret123"  # change later for security

# ===== USER (STATIC LOGIN) =====
USER = {
    "username": "admin",
    "password": "1234"
}

# ===== LOAD DATA =====
df = pd.read_csv('Book3.csv')

feature_names = df.drop(columns=['label']).columns.tolist()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df.drop(columns=['label']).values
y = df['label'].values

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ===== HISTORY =====
history = []

# ===== WEATHER =====
def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=20.3&longitude=85.8&current_weather=true"
        data = requests.get(url).json()
        return str(data['current_weather']['temperature']) + "°C"
    except:
        return "Unavailable"

# ===== LOGIN PAGE =====
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USER['username'] and password == USER['password']:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid Credentials")

    return render_template('login.html')

# ===== LOGOUT =====
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ===== HOME =====
@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template(
        'index.html',
        feature_names=feature_names,
        history=history,
        weather=get_weather()
    )

# ===== PREDICT =====
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        input_data = [float(request.form[name]) for name in feature_names]
        final_input = np.array([input_data])

        prediction_num = model.predict(final_input)[0]
        prediction = le.inverse_transform([prediction_num])[0]

        try:
            prob = model.predict_proba(final_input)[0][prediction_num]
        except:
            prob = 0.5

        history.append({
            "crop": prediction,
            "confidence": round(prob * 100, 2)
        })

        return render_template(
            'index.html',
            feature_names=feature_names,
            prediction_text=prediction,
            probability=round(prob * 100, 2),
            history=history,
            weather=get_weather()
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ===== CSV UPLOAD =====
@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        file = request.files['file']
        data = pd.read_csv(file)

        data = data[feature_names]

        preds = model.predict(data)
        crops = le.inverse_transform(preds)

        data['Prediction'] = crops

        return data.to_html(classes="table table-bordered table-striped")

    except Exception as e:
        return f"Error: {str(e)}"

# ===== CLEAR HISTORY =====
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history = []
    return redirect(url_for('home'))

# ===== DOWNLOAD CSV =====
@app.route('/download_history')
def download_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    df_hist = pd.DataFrame(history)

    if df_hist.empty:
        return "No history available"

    buffer = io.StringIO()
    df_hist.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='history.csv'
    )

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)