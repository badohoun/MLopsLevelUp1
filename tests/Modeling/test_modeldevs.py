from logging import info 
from numpy import isnan
import pytest
from customersatisfaction.Loading.load_data import  ingest_data
from customersatisfaction.Preprocessing.cleaning_data import DataCleaning  , DataDivideStrategy , DataPreprocessStrategy
from customersatisfaction.Modeling.modelsdevs import RandomForestModel , LightGBMModel, XGBoostModel,LinearRegressionModel , HyperparameterTuner


# Test pour la classe HyperparameterTuner
@pytest.mark.slow
def test_hyperparameter_tuner():
    try:
        data_frame = ingest_data()
        your_data_cleaner = DataCleaning(data_frame, DataPreprocessStrategy())
        preprocessed_data = your_data_cleaner.handle_data()
        your_data_divider = DataCleaning(preprocessed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = your_data_divider.handle_data()
        # Check for NaN values in the input data using Pandas functions
        assert not X_train.isna().any().any(),"There is na inside."
        assert not X_test.isna().any().any(),"There is na inside."
        assert not y_train.isna().any().any(),"There is na inside."
        assert not y_train.isna().any().any(),"There is na inside."
        # Test pour RandomForestModel
        tuner_rf = HyperparameterTuner(RandomForestModel(), X_train, y_train, X_test, y_test)
        best_params_rf = tuner_rf.optimize(n_trials=1)
        assert isinstance(best_params_rf, dict)
        # Test pour LightGBMModel
        tuner_lgbm = HyperparameterTuner(LightGBMModel(), X_train, y_train, X_test, y_test)
        best_params_lgbm = tuner_lgbm.optimize(n_trials=1)
        assert isinstance(best_params_lgbm, dict)
        # Test pour XGBoostModel
        tuner_xgb = HyperparameterTuner(XGBoostModel(), X_train, y_train, X_test, y_test)
        best_params_xgb = tuner_xgb.optimize(n_trials=1)
        assert isinstance(best_params_xgb, dict)
        # Test pour LinearRegressionModel
        tuner_lr = HyperparameterTuner(LinearRegressionModel(), X_train, y_train, X_test, y_test)
        best_params_lr = tuner_lr.optimize(n_trials=1)
        assert isinstance(best_params_lr, dict)
        info("models and phyperparameter devs Assertion test passed.")
    except Exception as e:
        pytest.fail(f"Test failed due to an exception: {e}")