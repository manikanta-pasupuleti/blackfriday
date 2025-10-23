import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Use a simple default secret for local dev; override with FLASK_SECRET in env
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-for-local')

# Directory where models are stored (project root /models)
if os.environ.get('RENDER'):
    MODELS_DIR = '/opt/render/project/src/models'
else:
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

logger.info(f"Using models directory: {MODELS_DIR}")
model_cache = {}

def load_model(name):
    """Load model by key (cached). Expects file {name}.pkl under MODELS_DIR."""
    try:
        if name in model_cache:
            logger.debug(f"Using cached model: {name}")
            return model_cache[name]

        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            available = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
            raise FileNotFoundError(
                f"Model file not found: {path}. Available models: {available}")

        logger.info(f"Loading model {name} from {path}")
        model = joblib.load(path)
        model_cache[name] = model
        return model
    except Exception as e:
        logger.error(f"Error loading model {name}: {e}")
        raise


# Preload common models to avoid slow first-request latency for large pickles
def preload_models(names=("rf_model",)):
    for n in names:
        try:
            load_model(n)
        except Exception as e:
            logger.error(f"Could not preload model {n}: {e}")

# call preload during import/startup
preload_models(("rf_model",))

# ------------------- Routes ------------------- #

@app.route('/')
def home():
    # Prefer prediction passed in query string (works even if client doesn't
    # accept cookies). Fall back to session-stored prediction and form data.
    # Do NOT trust the query string for showing predictions (avoids stale values
    # like ?prediction=381). Only show a session-stored prediction when there is
    # also stored form data from a real prediction request.
    form_data = session.get('last_form')
    pred = session.get('last_prediction') if form_data else None

    # Only construct a human-friendly prediction_text when we have a numeric
    # prediction value produced by the model. This avoids showing a stale or
    # default numeric value on first load.
    prediction_text = None
    if pred is not None:
        try:
            pred_val = int(float(pred))
            prediction_text = f"Predicted Purchase: ${pred_val:,.0f}"
        except Exception:
            # If pred is not numeric, show it as-is (e.g., an error message)
            prediction_text = str(pred)

    return render_template('index.html', prediction_text=prediction_text, form_data=form_data)


@app.route('/predict', methods=['POST'])
def predict():
    """HTML form prediction route."""
    data = request.form
    logger.info(f"Received form data: {dict(data)}")

    if not data:
        return redirect(url_for('home'))

    try:
        gender_map = {'M': 1, 'F': 0}
        age_map = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
        city_map = {'A': 0, 'B': 1, 'C': 2}

        gender = gender_map.get(data.get('Gender'), 0)
        age = age_map.get(data.get('Age'), 2)
        occupation = int(data.get('Occupation', 0))
        city = city_map.get(data.get('City_Category'), 0)

        stay_val = data.get('Stay_In_Current_City_Years', '0')
        stay = int(stay_val.replace('+', '')) if isinstance(stay_val, str) and stay_val.endswith('+') else int(stay_val)

        # Model expects 5 features: Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years
        features = np.array([[gender, age, occupation, city, stay]])
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

        # If the model was trained using a DataFrame, it may have feature names
        # recorded (feature_names_in_). Passing a DataFrame avoids sklearn's
        # "X does not have valid feature names" warning and is safer.
        try:
            fnames = getattr(model, 'feature_names_in_', None)
            if fnames is not None:
                cols = list(fnames[:features.shape[1]])
                features_df = pd.DataFrame(features, columns=cols)
                pred_val = model.predict(features_df)[0]
            else:
                pred_val = model.predict(features)[0]
        except Exception:
            # fallback to direct predict if anything goes wrong
            pred_val = model.predict(features)[0]

        pred = int(round(pred_val))
        logger.info(f"Prediction successful: {pred}")

    except Exception as e:
        logger.exception("Prediction failed")
        # Save a short error message to session so the UI can show it.
        session['last_error'] = str(e)
        return render_template('index.html', prediction_text=f"Error: {e}")

    # Render the result immediately (no session storage) to avoid persisting
    # a stale prediction value across requests. This makes the behavior
    # deterministic: a POST returns the predicted value, and refresh will
    # re-run the POST (PRG is not used in this variant).
    result_text = f"Predicted Purchase: ${pred:,.0f}"
    return render_template('index.html', prediction_text=result_text, form_data=dict(data))


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
        occupation = int(data.get('Occupation', 0))
        city = city_map.get(data.get('City_Category'), 0)

        stay_val = data.get('Stay_In_Current_City_Years', '0')
        stay = int(stay_val.replace('+', '')) if isinstance(stay_val, str) and stay_val.endswith('+') else int(stay_val)

        features = np.array([[gender, age, occupation, city, stay]])
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

        try:
            fnames = getattr(model, 'feature_names_in_', None)
            if fnames is not None:
                cols = list(fnames[:features.shape[1]])
                features_df = pd.DataFrame(features, columns=cols)
                pred_val = model.predict(features_df)[0]
            else:
                pred_val = model.predict(features)[0]
        except Exception:
            pred_val = model.predict(features)[0]

        pred = int(round(pred_val))

    except Exception as e:
        logger.exception("API prediction failed")
        session['last_error'] = str(e)
        return jsonify({'error': f'Prediction error: {e}'}), 500

    return jsonify({'prediction': pred})


@app.route('/clear')
def clear():
    """Clear stored prediction and form data from the session."""
    session.pop('last_prediction', None)
    session.pop('last_form', None)
    return redirect(url_for('home'))


@app.route('/health')
def health():
    """Simple health check for load balancers and platform probes."""
    return 'ok', 200



# ------------------- Main ------------------- #

if __name__ == '__main__':
    # Use PORT when provided by the environment (Render sets this).
    port = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', '5000')))
    host = '0.0.0.0' if (os.environ.get('RENDER') or 'PORT' in os.environ) else '127.0.0.1'
    debug_mode = os.environ.get('FLASK_DEBUG', '0') in ('1', 'true', 'True')
    use_reloader = debug_mode and not os.environ.get('RENDER')

    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    try:
        logger.info(f"Models directory: {MODELS_DIR}")
        logger.info(f"Available models: {[f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]}")
    except Exception:
        logger.warning("Could not list models directory contents")

    app.run(host=host, port=port, debug=debug_mode, use_reloader=use_reloader)
