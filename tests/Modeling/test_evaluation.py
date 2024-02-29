from logging import info
import pytest
from numpy import random
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from customersatisfaction.Loading.load_data import ingest_data
from customersatisfaction.Preprocessing.cleaning_data import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from customersatisfaction.Modeling.modelsdevs import (
    RandomForestModel,
    LightGBMModel,
    XGBoostModel,
    LinearRegressionModel,
    HyperparameterTuner,
)
from customersatisfaction.Modeling.evaluation import Evaluation, MSE, RMSE, R2Score


# MLS models metrics performance
@pytest.mark.slow
def test_metrics():
    try:
        data_frame = ingest_data()
        your_data_cleaner = DataCleaning(data_frame, DataPreprocessStrategy())
        preprocessed_data = your_data_cleaner.handle_data()
        your_data_divider = DataCleaning(preprocessed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = your_data_divider.handle_data()
        # Model training
        random_forest_model = RandomForestModel()
        lightgbm_model = LightGBMModel()
        xgboost_model = XGBoostModel()
        linear_regression_model = LinearRegressionModel()
        # Hyper parameter tuning
        tuner_rf = HyperparameterTuner(random_forest_model, X_train, X_test, y_train, y_test)
        tuner_lgbm = HyperparameterTuner(lightgbm_model, X_train, X_test, y_train, y_test)
        tuner_xgb = HyperparameterTuner(xgboost_model, X_train, X_test, y_train, y_test)
        tuner_lr = HyperparameterTuner(linear_regression_model, X_train, X_test, y_train, y_test)
        # Models  Loading
        rf_model = tuner_rf.model.train(x_train=X_train, y_train=y_train)
        lgbm_model = tuner_lgbm.model.train(x_train=X_train, y_train=y_train)
        xgb_model = tuner_xgb.model.train(x_train=X_train, y_train=y_train)
        lr_model = tuner_lr.model.train(x_train=X_train, y_train=y_train)
        # Models Prediction
        y_pred_rf = rf_model.predict(X_test)
        y_pred_lgbm = lgbm_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_lr = lr_model.predict(X_test)
        mse_evaluator = MSE()
        r2_evaluator = R2Score()
        mse_score_rf = mse_evaluator.calculate_score(y_test, y_pred_rf)
        r2_score_rf = r2_evaluator.calculate_score(y_test, y_pred_rf)
        assert isinstance(mse_score_rf, float)
        assert mse_score_rf >= 0  # La valeur MSE doit être non négative
        assert isinstance(r2_score_rf, float)
        assert -1 <= r2_score_rf <= 1  # La valeur R2 doit être entre -1 et 1 inclus
        mse_score_lgbm = mse_evaluator.calculate_score(y_test, y_pred_lgbm)
        r2_score_lgbm = r2_evaluator.calculate_score(y_test, y_pred_lgbm)
        assert isinstance(mse_score_lgbm, float)
        assert mse_score_lgbm >= 0  # La valeur MSE doit être non négative
        assert isinstance(r2_score_lgbm, float)
        assert -1 <= r2_score_lgbm <= 1  # La valeur R2 doit être entre -1 et 1 inclus
        mse_score_xgb = mse_evaluator.calculate_score(y_test, y_pred_xgb)
        r2_score_xgb = r2_evaluator.calculate_score(y_test, y_pred_xgb)
        assert isinstance(mse_score_xgb, float)
        assert mse_score_xgb >= 0  # La valeur MSE doit être non négative
        assert isinstance(r2_score_xgb, float)
        assert -1 <= r2_score_xgb <= 1  # La valeur R2 doit être entre -1 et 1 inclus
        mse_score_lr = mse_evaluator.calculate_score(y_test, y_pred_lr)
        r2_score_lr = r2_evaluator.calculate_score(y_test, y_pred_lr)
        assert isinstance(mse_score_lr, float)
        assert mse_score_lr >= 0  # La valeur MSE doit être non négative
        assert isinstance(r2_score_lr, float)
        assert -1 <= r2_score_lr <= 1  # La valeur R2 doit être entre -1 et 1 inclus
        info("Metrics Assertion test passed.")
    except Exception as e:
        pytest.fail(e)
