# backend/app.py
def load_model(name):
if name in model_cache:
return model_cache[name]
path = os.path.abspath(os.path.join(MODELS_DIR, f"{name}.pkl"))
model = joblib.load(path)
model_cache[name] = model
return model


@app.route('/')
def home():
return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
# Expected form fields: model_choice, Gender, Age, City_Category, Stay_In_Current_City_Years,
# Marital_Status, Product_Category_1, Product_Category_2, Product_Category_3
data = request.form
# Map inputs to integers/floats
try:
gender = int(data.get('Gender'))
age = int(data.get('Age'))
city = int(data.get('City_Category'))
stay = int(data.get('Stay_In_Current_City_Years'))
marital = int(data.get('Marital_Status'))
p1 = float(data.get('Product_Category_1', 0))
p2 = float(data.get('Product_Category_2', 0))
p3 = float(data.get('Product_Category_3', 0))
except Exception as e:
return render_template('index.html', prediction_text=f"Invalid input: {e}")


features = np.array([[gender, age, city, stay, marital, p1, p2, p3]])


choice = data.get('model_choice', 'rf_model')
# map choice names to filenames used in models/
mapping = {
'rf': 'rf_model',
'ab': 'ab_model',
'gb': 'gb_model',
'lr': 'lr_model',
'dt': 'dt_model',
'knn': 'knn_model'
}
model_key = mapping.get(choice, 'rf_model')


model = load_model(model_key)
pred = model.predict(features)[0]
pred = int(pred)


return render_template('index.html', prediction_text=f"Predicted Purchase: {pred}")


if __name__ == '__main__':
app.run(debug=True)