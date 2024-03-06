import pandas as pd
from flask import Flask, request, jsonify
from customersatisfaction.Loading.load_data import ingest_data
from customersatisfaction.Preprocessing.cleaning_data import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from customersatisfaction.Modeling.modelsdevs import (
    RandomForestModel,
    LightGBMModel,
    XGBoostModel,
    LinearRegressionModel,
    HyperparameterTuner,
)
from customersatisfaction.Modeling.modelspredicts import load_artifacts



my_flask_app = Flask(__name__)



@my_flask_app.route('/', methods=['GET'])
def route_home():
    return "OK !", 200

@my_flask_app.route('/predict', methods=['POST'])
def route_predict():
    try:
        body = request.get_json()
        df = pd.DataFrame.from_dict(body)
        your_data_cleaner = DataCleaning(df, DataPreprocessStrategy())
        preprocessed_data = your_data_cleaner.handle_data()
        preprocessed_data.drop(["review_score"], axis=1, inplace=True)

        results = {'customersatisfactionscore': list(load_artifacts().predict(df).flatten())}
        return jsonify(results), 200
    except Exception as e:
        logging.error(e)
        raise e




if __name__ == "__main__":
    my_flask_app.run(port=7000)
