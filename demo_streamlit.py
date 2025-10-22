import streamlit as st
import requests

st.set_page_config(page_title="Black Friday Predictor", layout="centered")

st.title("Black Friday Purchase Prediction (Demo)")

with st.form('predict'):
	model_choice = st.selectbox('Model', ['rf', 'ab', 'gb', 'lr', 'dt', 'knn'], index=0)
	user_id = st.text_input('User ID', '1000001')
	product_id = st.text_input('Product ID', 'P00069042')
	gender = st.selectbox('Gender', ['M', 'F'])
	age = st.selectbox('Age', ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], index=2)
	occupation = st.number_input('Occupation', min_value=0, value=10)
	city = st.selectbox('City Category', ['A', 'B', 'C'])
	stay = st.selectbox('Stay In Current City Years', ['0', '1', '2', '3', '4+'], index=2)
	marital = st.selectbox('Marital Status', [0, 1], index=0)
	p1 = st.number_input('Product Category 1', min_value=0, value=3)
	p2 = st.number_input('Product Category 2', min_value=0, value=0)
	p3 = st.number_input('Product Category 3', min_value=0, value=0)
	submitted = st.form_submit_button('Predict')

if submitted:
	payload = {
		'model_choice': model_choice,
		'User_ID': user_id,
		'Product_ID': product_id,
		'Gender': gender,
		'Age': age,
		'Occupation': occupation,
		'City_Category': city,
		'Stay_In_Current_City_Years': stay,
		'Marital_Status': marital,
		'Product_Category_1': p1,
		'Product_Category_2': p2,
		'Product_Category_3': p3,
	}

	try:
		r = requests.post('http://127.0.0.1:5000/api/predict', json=payload, timeout=10)
		if r.status_code == 200:
			result = r.json()
			st.success(f"Predicted Purchase: {result.get('prediction')}")
		else:
			st.error(f"API error {r.status_code}: {r.text}")
	except Exception as e:
		st.error(f"Request failed: {e}")
