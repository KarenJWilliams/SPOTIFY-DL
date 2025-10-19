from flask import Flask, request, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow import keras
import xgboost as xgb
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# =========================
# Load trained models & scaler
# =========================
fnn_model_path = 'models/fnn_model.keras'
xgb_model_path = 'models/xgb_model.pkl'
scaler_path = 'models/scaler.pkl'

if not os.path.exists(fnn_model_path) or not os.path.exists(xgb_model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Make sure all models and scaler exist in 'models/' directory.")

fnn_model = keras.models.load_model(fnn_model_path)
xgb_model = joblib.load(xgb_model_path)
scaler = joblib.load(scaler_path)

# =========================
# Load dataset for recommendations
# =========================
dataset_path = "dataset.csv"
df_full = pd.read_csv(dataset_path)
metadata_cols = ['track_name', 'artists', 'album_name', 'track_id']
metadata = df_full[metadata_cols].copy()
df_features = df_full.drop(columns=metadata_cols + ['index'], errors='ignore')
df_features = pd.get_dummies(df_features, columns=['track_genre', 'explicit'], drop_first=True)

feature_columns = scaler.feature_names_in_
for col in feature_columns:
    if col not in df_features.columns:
        df_features[col] = 0
df_features = df_features[feature_columns]
df_features_scaled = scaler.transform(df_features)

# =========================
# HTML template with CSS animations
# =========================
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spotify Popularity Predictor & Recommender</title>
<style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #1DB954, #191414);
        color: #fff;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        min-height: 100vh;
        flex-direction: column;
        animation: gradient 10s ease infinite;
        background-size: 400% 400%;
    }

    @keyframes gradient {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    h1 {
        text-align: center;
        margin-top: 30px;
        text-shadow: 2px 2px 5px #000;
    }

    form {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background: rgba(0,0,0,0.6);
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        animation: fadeIn 2s ease;
    }

    label {
        font-weight: bold;
    }

    input[type=number] {
        padding: 8px;
        border-radius: 8px;
        border: none;
        outline: none;
        width: 100%;
        box-sizing: border-box;
        transition: transform 0.2s;
    }

    input[type=number]:focus {
        transform: scale(1.05);
        box-shadow: 0 0 10px #1DB954;
    }

    button {
        grid-column: span 2;
        padding: 15px;
        font-size: 16px;
        background: #1DB954;
        color: #fff;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.2s, background 0.3s;
    }

    button:hover {
        transform: scale(1.05);
        background: #1ed760;
    }

    .results {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background: rgba(0,0,0,0.6);
        border-radius: 15px;
        animation: fadeIn 2s ease;
    }

    .recommendation-card {
        background: rgba(29,185,84,0.2);
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        transition: transform 0.2s;
    }

    .recommendation-card:hover {
        transform: scale(1.02);
        background: rgba(29,185,84,0.4);
    }

    @keyframes fadeIn {
        from {opacity:0;}
        to {opacity:1;}
    }

    pre {
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 10px;
        overflow-x: auto;
    }
</style>
</head>
<body>
<h1>Spotify Popularity Predictor & Recommender</h1>
<form method="post" action="/submit">
  {% for col in feature_columns %}
    <label>{{ col }}:</label>
    <input type="number" step="any" name="{{ col }}" value="0">
  {% endfor %}
  <button type="submit">Predict & Recommend</button>
</form>

{% if result %}
<div class="results">
    <h2>Prediction Result</h2>
    <pre>{{ result['prediction'] }}</pre>

    <h2>Top Recommendations</h2>
    {% for rec in result['recommendations'] %}
        <div class="recommendation-card">
            <strong>{{ rec['track_name'] }}</strong> by {{ rec['artists'] }}<br>
            Album: {{ rec['album_name'] }}<br>
            Predicted Popularity: {{ rec['predicted_popularity'] | round(2) }}
        </div>
    {% endfor %}
</div>
{% endif %}
</body>
</html>
"""

# =========================
# Home route
# =========================
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, feature_columns=feature_columns, result=None)

# =========================
# Form submission
# =========================
@app.route('/submit', methods=['POST'])
def submit():
    data = {col: float(request.form.get(col, 0)) for col in feature_columns}
    features = pd.DataFrame([data])
    features_scaled = scaler.transform(features)

    fnn_pred = fnn_model.predict(features_scaled).flatten()
    xgb_pred = xgb_model.predict(features_scaled)
    final_pred = (fnn_pred + xgb_pred) / 2

    prediction = {
        'fnn_prediction': float(fnn_pred[0]),
        'xgb_prediction': float(xgb_pred[0]),
        'popularity_prediction': float(final_pred[0])
    }

    # Recommendations
    similarities = cosine_similarity(features_scaled, df_features_scaled)
    top_indices = similarities[0].argsort()[::-1][1:6]
    recommendations = []
    for idx in top_indices:
        rec = metadata.iloc[idx].to_dict()
        rec_features = df_features_scaled[idx:idx+1]
        rec['predicted_popularity'] = float((fnn_model.predict(rec_features) + xgb_model.predict(rec_features)) / 2)
        recommendations.append(rec)

    result = {'prediction': prediction, 'recommendations': recommendations}
    return render_template_string(HTML_TEMPLATE, feature_columns=feature_columns, result=result)

# =========================
# Run Flask
# =========================
if __name__ == '__main__':
    app.run(debug=True)
