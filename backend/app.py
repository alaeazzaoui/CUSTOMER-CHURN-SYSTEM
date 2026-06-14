from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Charger les modèles
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('cat_model.pkl', 'rb') as f:
    cat_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_choice = data.get('model', 'rf')
    input_data = data.get('features', {})

    # Créer DataFrame
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # Normaliser
    df_scaled = scaler.transform(df)

    # Choisir le modèle
    if model_choice == 'rf':
        model = rf_model
    elif model_choice == 'xgb':
        model = xgb_model
    else:
        model = cat_model

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return jsonify({
        'prediction': int(prediction),
        'probability': round(float(probability) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)