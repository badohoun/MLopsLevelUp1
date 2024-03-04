def test_predict_endpoint():
        try:
            # Assuming your Flask app is running locally on port 7000
            api_url = 'http://127.0.0.1:7000/predict'
    
            # Define a sample input data
            input_data = {
                "payment_sequential": 1,
                "payment_installments": 0,
                "payment_value": 35.50,
                "price": 29.99,
                "freight_value": 8.72,
                "product_name_lenght": 40.0,
                "product_description_lenght": 4.0,
                "product_photos_qty": 500.0,
                "product_weight_g": 19.0,
                "product_length_cm": 8.0,
                "product_height_cm": 13.0,
                "product_width_cm": 9.0
            }
    
            app = Flask(__name__)
    
            with app.app_context():
                # Make a POST request to the /predict endpoint
                response = requests.post(api_url, json=input_data)
    
                # Check if the response status code is 200 (OK)
                assert response.status_code == 200
    
                # Parse the JSON response
                result = response.json()
    
                # Check if the expected key is present in the result
                assert 'customersatisfactionscore' in result
    
                # Optionally, you can check specific conditions on the output
                # For example, if you expect the score to be a float between 0 and 1:
                score = result['customersatisfactionscore'][0]
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1
                print("Api assertion test passed.")
        except Exception as e:
             return jsonify({"error": str(e)}), 500
