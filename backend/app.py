import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Directory where models are stored (project root /models)
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_cache = {}

def load_model(name):
    """Load model by key (cached). Expects file {name}.pkl under MODELS_DIR."""
    if name in model_cache:
        return model_cache[name]
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    model_cache[name] = model
    return model

# ------------------- Routes ------------------- #

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """HTML form prediction route."""
    data = request.form
    try:
        gender_map = {'M': 1, 'F': 0}
        age_map = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
        city_map = {'A': 0, 'B': 1, 'C': 2}

        gender = gender_map.get(data.get('Gender'), 0)
        age = age_map.get(data.get('Age'), 2)
        city = city_map.get(data.get('City_Category'), 0)

        stay_val = data.get('Stay_In_Current_City_Years', '0')
        stay = int(stay_val.replace('+', '')) if stay_val.endswith('+') else int(stay_val)

        marital = int(data.get('Marital_Status', 0))
        p1 = float(data.get('Product_Category_1', 0) or 0)
        p2 = float(data.get('Product_Category_2', 0) or 0)
        p3 = float(data.get('Product_Category_3', 0) or 0)

        features = np.array([[gender, age, city, stay, marital, p1, p2, p3]])
        choice = data.get('model_choice', 'rf')

        model_key = {
            'rf': 'rf_model',
            'ab': 'ab_model',
            'gb': 'gb_model',
            'lr': 'lr_model',
            'dt': 'dt_model',
            'knn': 'knn_model'
        }.get(choice, 'rf_model')

        model = load_model(model_key)
        pred = int(round(model.predict(features)[0]))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

    return render_template('index.html', prediction_text=f"Predicted Purchase: {pred}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API route."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Expected JSON body'}), 400

    try:
        gender_map = {'M': 1, 'F': 0}
        age_map = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
        city_map = {'A': 0, 'B': 1, 'C': 2}

        gender = gender_map.get(data.get('Gender'), 0)
        age = age_map.get(data.get('Age'), 2)
        city = city_map.get(data.get('City_Category'), 0)

        stay_val = data.get('Stay_In_Current_City_Years', '0')
        stay = int(stay_val.replace('+', '')) if isinstance(stay_val, str) and stay_val.endswith('+') else int(stay_val)

        marital = int(data.get('Marital_Status', 0))
        p1 = float(data.get('Product_Category_1', 0) or 0)
        p2 = float(data.get('Product_Category_2', 0) or 0)
        p3 = float(data.get('Product_Category_3', 0) or 0)

        features = np.array([[gender, age, city, stay, marital, p1, p2, p3]])
        choice = data.get('model_choice', 'rf')

        model_key = {
            'rf': 'rf_model',
            'ab': 'ab_model',
            'gb': 'gb_model',
            'lr': 'lr_model',
            'dt': 'dt_model',
            'knn': 'knn_model'
        }.get(choice, 'rf_model')

        model = load_model(model_key)

        # Handle feature mismatch (if any)
        expected = getattr(model, 'n_features_in_', None)
        if expected and expected != features.shape[1]:
            adj = np.zeros((1, expected))
            adj[0, :min(expected, features.shape[1])] = features[0, :min(expected, features.shape[1])]
            features = adj

        pred = int(round(model.predict(features)[0]))

    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

    return jsonify({'prediction': pred})


# ------------------- Main ------------------- #

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '1') in ('1', 'true', 'True')
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    print(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
