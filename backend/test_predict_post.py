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
    with requests.Session() as s:
        # POST and allow redirects; session preserves cookies so server-side
        # session data is available on the redirected GET.
        r = s.post(url, data=data, timeout=120)
        print('POST final status', r.status_code)
        # r is the response after following redirects (requests follows them by default)
        print('Final URL after redirects:', r.url)
        if 'Result:' in r.text:
            idx = r.text.find('Result:')
            print('Found Result at', idx)
            print(r.text[idx-80:idx+200])
        else:
            print('Result not found in final response (length', len(r.text), ')')
            # as a fallback, do an explicit GET on r.url (should be same)
            r2 = s.get(r.url, timeout=10)
            print('GET status', r2.status_code, 'Result in GET?', 'Result:' in r2.text)
            if 'Result:' in r2.text:
                i = r2.text.find('Result:')
                print(r2.text[i-80:i+200])
except Exception as e:
    print('Request failed:', e)
