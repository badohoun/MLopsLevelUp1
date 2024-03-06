import pytest
import requests
import pytest
import requests
from app import my_flask_app

@pytest.fixture
def api_url():
    return 'http://127.0.0.1:7000'  # Update the URL if your API is running on a different port

def test_home(api_url):
    response = requests.get(f'{api_url}/')
    assert response.status_code == 200
    assert response.text == "OK !"

def test_predict_endpoint(api_url):
    input_data = {
        "payment_sequential": 1,
        "payment_installments": 0,
        "payment_value": 35.50,
        "price": 29.99,
        "freight_value": 8.72,
        "product_name_length": 40.0,
        "product_description_length": 4.0,
        "product_photos_qty": 500.0,
        "product_weight_g": 19.0,
        "product_length_cm": 8.0,
        "product_height_cm": 13.0,
        "product_width_cm": 9.0
    }

    response = requests.post(f'{api_url}/predict', json=input_data)
    assert response.status_code == 200

    result = response.json()
    assert 'customersatisfactionscore' in result
    score = result['customersatisfactionscore'][0]
    assert isinstance(score, (int, float))
    assert 0 <= score <= 1

