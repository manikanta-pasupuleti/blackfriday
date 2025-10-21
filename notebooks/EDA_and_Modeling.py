import pandas as pd
print('\nRunning GridSearch for RandomForest...')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print('Best RF params:', grid_rf.best_params_)


# AdaBoost
param_ab = {
'n_estimators': [50, 100, 200],
'learning_rate': [0.01, 0.1, 1]
}


grid_ab = GridSearchCV(AdaBoostRegressor(random_state=42), param_ab, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
print('\nRunning GridSearch for AdaBoost...')
grid_ab.fit(X_train, y_train)
best_ab = grid_ab.best_estimator_
print('Best AB params:', grid_ab.best_params_)


# Gradient Boosting
param_gb = {
'n_estimators': [100, 200],
'learning_rate': [0.05, 0.1],
'max_depth': [3, 5]
}


grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_gb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
print('\nRunning GridSearch for GradientBoosting...')
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_
print('Best GB params:', grid_gb.best_params_)


# 7. Evaluate tuned ensemble models
for name, model in [('RandomForest', best_rf), ('AdaBoost', best_ab), ('GradientBoosting', best_gb)]:
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\n{name} tuned --> MSE: {mse:.2f}, R2: {r2:.4f}")


# 8. Save models to models/ folder
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(best_rf, '../models/rf_model.pkl')
joblib.dump(best_ab, '../models/ab_model.pkl')
joblib.dump(best_gb, '../models/gb_model.pkl')
# Save the basic models too (baseline)
joblib.dump(models['LinearRegression'], '../models/lr_model.pkl')
joblib.dump(models['DecisionTree'], '../models/dt_model.pkl')
joblib.dump(models['KNN'], '../models/knn_model.pkl')
print('\nSaved all models to ../models/')


# 9. (Optional) Simple plots comparing model MSEs
try:
import matplotlib.pyplot as plt
names = list(results.keys())
mses = [results[n]['mse'] for n in names]
plt.figure(figsize=(10,5))
plt.bar(names, mses)
plt.title('Baseline models MSE (no tuning)')
plt.ylabel('MSE')
plt.show()
except Exception as e:
print('Plot skipped (not running in interactive mode).', e)


print('\nAll done!')