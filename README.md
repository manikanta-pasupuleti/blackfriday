
# Black Friday Purchase Prediction

This repository contains code and artifacts for a small web app that demonstrates purchase prediction on the Black Friday dataset. It includes a Flask backend (in `backend/`) and a Streamlit demo (`demo_streamlit.py`). The original dataset (a cleaned subset of a public Black Friday dataset) is in `data/BlackFriday.csv`.

## Dataset (data/BlackFriday.csv)

Top of file (header and first 5 rows):

User_ID,Product_ID,Gender,Age,Occupation,City_Category,Stay_In_Current_City_Years,Marital_Status,Product_Category_1,Product_Category_2,Product_Category_3,Purchase
1000001,P00069042,F,0-17,10,A,2,0,3,,,8370
1000001,P00248942,F,0-17,10,A,2,0,1,6,14,15200
1000001,P00087842,F,0-17,10,A,2,0,12,,,1422
1000001,P00085442,F,0-17,10,A,2,0,12,14,,1057
1000002,P00285442,M,55+,16,C,4+,0,8,,,7969

Column descriptions (inferred):
- User_ID: unique identifier for the user
- Product_ID: unique product code
- Gender: M/F
- Age: age bracket (e.g. '0-17', '18-25', ..., '55+')
- Occupation: encoded occupation id
- City_Category: city category A/B/C
- Stay_In_Current_City_Years: number of years in current city ('4+' appears)
- Marital_Status: 0/1 indicator
- Product_Category_1/2/3: product category encodings (some values missing)
- Purchase: numeric purchase amount (target)

Dataset size: ~550k rows (file contains 550,070 lines).

## Requirements

Top-level `requirements.txt` and `backend/requirements.txt` are provided. Key packages used:
- pandas, numpy
- scikit-learn
- joblib (for loading models)
- flask (backend)
- streamlit (optional demo)

Install dependencies into a virtual environment (Windows PowerShell example):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or install backend-only dependencies:

```powershell
pip install -r backend/requirements.txt
```

If you plan to deploy with Gunicorn (Linux servers), use the `backend/requirements.txt` which includes `gunicorn`.

## Running the Flask backend

From the project root you can start the backend that serves the HTML prediction page located in `backend/templates/index.html`:

```powershell
python backend/app.py
```

By default the app binds to `127.0.0.1:5000`. The `MODELS_DIR` defaults to `../models` relative to `backend/` — make sure any trained `.pkl` model files are placed under `models/` or point `MODELS_DIR` via the `RENDER` env var for render-specific paths.

If you want a convenience top-level entry point (so `python app.py` from repo root works), create a thin wrapper that imports `backend.app` and runs it (not included by default).

## Running the Streamlit demo

```powershell
streamlit run demo_streamlit.py
```

## Notes
- The project expects pre-trained model pickles under `models/` with names like `rf_model.pkl`. If they are not present the app will log available models and fail to load the requested model.
- Missing values in product category columns are expected and handled during model preprocessing (check `notebooks/EDA_and_Modeling.py` for training code).

## Next steps / Troubleshooting
- If `python backend/app.py` raises a FileNotFoundError listing available models: either train and export models to `models/` or update `MODELS_DIR` environment variable.
- To reproduce errors reported when running `python app.py` from project root, run `python backend/app.py` directly to ensure correct working directory and relative path resolution.

If you'd like, I can:
- add a top-level `app.py` wrapper so `python app.py` works from the repo root,
- implement a small script that verifies models exist and prints helpful diagnostics,
- or try to run the backend here and capture the current error traceback.
