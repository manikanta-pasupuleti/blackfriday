"""EDA and modeling pipeline for Black Friday dataset.

This script was adapted to be self-contained so the later GridSearch
and model-saving steps run without NameError. It will:
 - load data from ../data/BlackFriday.csv
 - do minimal preprocessing
 - split into train/test
 - train a few baseline regressors and record results
 - run GridSearchCV for ensemble regressors
 - save tuned and baseline models to ../models

Assumptions:
 - target column is named 'Purchase'
"""

import os
import sys
import pickle

# Prefer joblib if available (faster for large numpy arrays); fall back to pickle
try:
    import joblib
    def save_model(obj, path):
        joblib.dump(obj, path)
except Exception:
    def save_model(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
def load_and_preprocess(path):
    """Load CSV, basic cleaning, encoding, and scaling. Returns X, y, scaler."""
    # pandas imported locally to avoid slow global import for --dry runs
    import pandas as pd
    df = pd.read_csv(path)
    # Basic check for target
    if 'Purchase' not in df.columns:
        raise KeyError("Expected target column 'Purchase' not found in the dataset.")

    # If dataset is very large, sample a subset for faster processing and
    # avoid expensive global operations (drop_duplicates/factorize) that can
    # be extremely slow on hundreds of thousands of rows.
    total_rows = len(df)
    if total_rows > 100000:
        print(f"Large dataset detected ({total_rows} rows). Sampling 50,000 rows for faster processing.")
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    else:
        # For smaller datasets it's OK to drop duplicates
        df = df.drop_duplicates()

    # Simple NA handling: fill numerical with median, categorical with mode
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Identify obvious identifier columns (contain 'id') and drop them to avoid
    # extremely high-cardinality one-hot encodings (e.g., user/product ids)
    id_candidates = [c for c in df.columns if 'id' in c.lower()]
    for c in id_candidates:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    if id_candidates:
        df = df.drop(columns=id_candidates)

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Separate features and target
    y = df['Purchase']
    X = df.drop(columns=['Purchase'])

    # For categorical columns, avoid full one-hot for very high-cardinality cols;
    # frequency-encode those instead and one-hot the low-cardinality ones.
    n_rows = len(df)
    high_card_threshold = min(100, int(0.5 * n_rows))
    high_card = [c for c in cat_cols if c in X.columns and X[c].nunique() > high_card_threshold]
    low_card = [c for c in cat_cols if c in X.columns and c not in high_card]

    # Frequency encode high-cardinality categoricals
    for c in high_card:
        freq = X[c].value_counts(normalize=True)
        X[c + '_freq'] = X[c].map(freq).fillna(0.0)

    # One-hot encode only the low-cardinality categoricals
    if low_card:
        X = pd.get_dummies(X, columns=low_card, drop_first=True)

    # Drop original high-card columns after frequency encoding
    X = X.drop(columns=high_card, errors='ignore')

    # Now all features should be numeric (freq columns + existing numeric + dummies)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Scale numeric features
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print('\nscikit-learn is required for preprocessing (StandardScaler).')
        print('Install it with: python -m pip install scikit-learn')
        raise

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    X = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

    return X, y, scaler


def train_baselines(X_train, y_train, X_test, y_test):
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error, r2_score
    except Exception:
        print('\nscikit-learn is required to train baseline models.')
        print('Install it with: python -m pip install scikit-learn')
        raise

    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor()
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'mse': mse, 'r2': r2}

    return models, results


def dry_run_quick_metrics(csv_path, sample_n=2000):
    """Very fast baseline using only Python stdlib: constant predictor (train mean).
    Reads up to sample_n rows from csv_path and computes MSE and R2 on a train/test split.
    This avoids pandas/numpy/sklearn imports so it starts immediately and returns quick feedback."""
    import csv
    from math import sqrt

    purchases = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        target_idx = None
        try:
            target_idx = header.index('Purchase')
        except ValueError:
            print("'Purchase' column not found in CSV header. Dry-run aborted.")
            return

        for i, row in enumerate(reader):
            if i >= sample_n:
                break
            # guard against malformed rows
            if len(row) <= target_idx:
                continue
            try:
                val = float(row[target_idx])
            except Exception:
                continue
            purchases.append(val)

    if len(purchases) < 10:
        print('Not enough rows for dry run; need at least 10 valid rows.')
        return

    # simple 80/20 split
    split = int(0.8 * len(purchases))
    train = purchases[:split]
    test = purchases[split:]

    mean_train = sum(train) / len(train)

    # compute MSE and R2 manually
    mse = sum((y - mean_train) ** 2 for y in test) / len(test)
    # R2 = 1 - SS_res / SS_tot
    mean_test = sum(test) / len(test)
    ss_res = sum((y - mean_train) ** 2 for y in test)
    ss_tot = sum((y - mean_test) ** 2 for y in test)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print('\nDRY RUN (fast) results on', len(purchases), 'rows (train/test=', len(train), '/', len(test), ')')
    print(f'Baseline constant predictor (train mean = {mean_train:.2f})')
    print(f'MSE: {mse:.2f}, R2: {r2:.4f}')



def main():
    # Resolve data path relative to this script file to avoid issues when
    # running from different working directories.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'BlackFriday.csv'))
    print('Loading data from', data_path)
    if not os.path.exists(data_path):
        print('\nError: dataset not found at', data_path)
        print('Please ensure the file exists. Example path relative to project root: data\\BlackFriday.csv')
        raise SystemExit(1)

    X, y, scaler = load_and_preprocess(data_path)

    # Train/test split (import locally so missing sklearn gives a helpful message)
    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        print('\nscikit-learn is required to perform train/test split.')
        print('Install it with: python -m pip install scikit-learn')
        raise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train baseline models
    print('\nTraining baseline models...')
    models, results = train_baselines(X_train, y_train, X_test, y_test)
    for name, res in results.items():
        print(f"{name} --> MSE: {res['mse']:.2f}, R2: {res['r2']:.4f}")

    # 6. Faster randomized search for ensemble models on a training subset
    try:
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, r2_score
    except Exception:
        print('\nscikit-learn (sklearn) is required for randomized search and ensemble models.')
        print('Install it with: python -m pip install scikit-learn')
        raise

    # Fast mode: use --fast on the command line to drastically reduce runtime
    fast = ('--fast' in sys.argv) or ('-f' in sys.argv)
    if fast:
        print('FAST MODE enabled: using very small sample and few iterations for quick feedback')

    # Use a smaller subset for hyperparameter search to save time/memory
    if fast:
        search_sample_n = min(2000, len(X_train))
    else:
        search_sample_n = min(10000, len(X_train))

    if search_sample_n < len(X_train):
        X_search = X_train.sample(n=search_sample_n, random_state=42)
        y_search = y_train.loc[X_search.index]
    else:
        X_search = X_train
        y_search = y_train

    # We'll use RandomizedSearchCV with few iterations and fewer CV folds
    n_iter_search = 2 if fast else 8
    cv_folds = 2
    print(f"\nRunning RandomizedSearch (n_iter={n_iter_search}, cv={cv_folds}) on {len(X_search)} rows")

    # Random Forest (randomized)
    param_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    rand_rf = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                 param_distributions=param_rf,
                                 n_iter=n_iter_search, cv=cv_folds,
                                 scoring='neg_mean_squared_error', n_jobs=1, verbose=2,
                                 random_state=42)
    rand_rf.fit(X_search, y_search)
    best_rf = rand_rf.best_estimator_
    print('Best RF params:', rand_rf.best_params_)

    # AdaBoost (randomized)
    param_ab = {
        'n_estimators': [30, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
    }
    rand_ab = RandomizedSearchCV(AdaBoostRegressor(random_state=42),
                                 param_distributions=param_ab,
                                 n_iter=n_iter_search, cv=cv_folds,
                                 scoring='neg_mean_squared_error', n_jobs=1, verbose=2,
                                 random_state=42)
    print('\nRunning RandomizedSearch for AdaBoost...')
    rand_ab.fit(X_search, y_search)
    best_ab = rand_ab.best_estimator_
    print('Best AB params:', rand_ab.best_params_)

    # Gradient Boosting (randomized)
    param_gb = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4]
    }
    rand_gb = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                                 param_distributions=param_gb,
                                 n_iter=n_iter_search, cv=cv_folds,
                                 scoring='neg_mean_squared_error', n_jobs=1, verbose=2,
                                 random_state=42)
    print('\nRunning RandomizedSearch for GradientBoosting...')
    rand_gb.fit(X_search, y_search)
    best_gb = rand_gb.best_estimator_
    print('Best GB params:', rand_gb.best_params_)

    # 7. Evaluate Tuned Ensemble Models
    print("\nEvaluating Tuned Ensemble Models...")
    for name, model in [('RandomForest', best_rf),
                        ('AdaBoost', best_ab),
                        ('GradientBoosting', best_gb)]:
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"\n{name} tuned --> MSE: {mse:.2f}, R2: {r2:.4f}")

    # 8. Save Models to 'models/' Folder (repo-root relative)
    repo_root = os.path.dirname(script_dir)
    out_dir = os.path.join(repo_root, 'models')
    os.makedirs(out_dir, exist_ok=True)

    save_model(best_rf, os.path.join(out_dir, 'rf_model.pkl'))
    save_model(best_ab, os.path.join(out_dir, 'ab_model.pkl'))
    save_model(best_gb, os.path.join(out_dir, 'gb_model.pkl'))

    # Save baseline models
    save_model(models['LinearRegression'], os.path.join(out_dir, 'lr_model.pkl'))
    save_model(models['DecisionTree'], os.path.join(out_dir, 'dt_model.pkl'))
    save_model(models['KNN'], os.path.join(out_dir, 'knn_model.pkl'))

    print('\n✅ Saved all models to', out_dir)

    # 9. (Optional) Simple Plot Comparing Baseline Model MSEs
    try:
        import matplotlib.pyplot as plt
        names = list(results.keys())
        mses = [results[n]['mse'] for n in names]

        plt.figure(figsize=(10, 5))
        plt.bar(names, mses, color='skyblue', edgecolor='black')
        plt.title('Baseline Models MSE (No Tuning)', fontsize=14)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except ModuleNotFoundError:
        print('matplotlib not installed; skipping plot.')
    except Exception as e:
        print('Plot skipped (not running in interactive mode).', e)

    print('\n🎯 All done!')


if __name__ == '__main__':
    # Fast dry-run option: use only stdlib CSV parsing and a constant predictor
    if '--dry' in sys.argv:
        # path relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'BlackFriday.csv'))
        dry_run_quick_metrics(data_path, sample_n=2000)
    else:
        main()
