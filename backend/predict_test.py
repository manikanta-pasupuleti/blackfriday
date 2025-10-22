import requests

url = 'http://127.0.0.1:5000/predict'
# sample data matching the form fields in index.html
data = {
    'model_choice': 'rf',
    'User_ID': '1000001',
    'Product_ID': 'P00069042',
    'Gender': 'M',
    'Age': '26-35',
    'Occupation': '10',
    'City_Category': 'A',
    'Stay_In_Current_City_Years': '2',
    'Marital_Status': '0',
    'Product_Category_1': '3',
    'Product_Category_2': '',
    'Product_Category_3': ''
}

try:
    r = requests.post(url, data=data, timeout=15)
    print('Status:', r.status_code)
    text = r.text
    # print a snippet of the response
    print(text[:4000])
except Exception as e:
    print('Request failed:', e)
