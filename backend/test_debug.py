import requests

url = 'http://127.0.0.1:5000/predict'
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
    r = requests.post(url, data=data, timeout=120)
    print('POST status:', r.status_code)
    
    # Check if Error is in response
    if 'Error:' in r.text:
        start = r.text.find('Error:')
        snippet = r.text[start:start+300]
        print('ERROR FOUND:')
        print(snippet)
    elif 'Result:' in r.text:
        start = r.text.find('Result:')
        snippet = r.text[start:start+200]
        print('RESULT FOUND:')
        print(snippet)
    else:
        print('Neither Result nor Error found in response')
        print('Response length:', len(r.text))
        # Print first 500 chars to debug
        print('First 500 chars:', r.text[:500])
        
except Exception as e:
    print('Request failed:', e)
