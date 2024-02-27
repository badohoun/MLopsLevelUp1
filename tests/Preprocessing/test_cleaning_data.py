from logging import info 
import pytest
from customersatisfaction.Loading.load_data import  ingest_data
from customersatisfaction.Preprocessing.cleaning_data import DataCleaning  , DataDivideStrategy , DataPreprocessStrategy



def test_prep_step():
    """Test the shape of the data after the data cleaning step."""
    try:
        data_frame = ingest_data()
        your_data_cleaner = DataCleaning(data_frame, DataPreprocessStrategy())
        preprocessed_data = your_data_cleaner.handle_data()
        your_data_divider = DataCleaning(preprocessed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = your_data_divider.handle_data()

        assert X_train.shape == (
            92487,
            12,
        ), "The shape of the training set is not correct."
        assert y_train.shape == (
            92487,
        ), "The shape of labels of training set is not correct."
        assert X_test.shape == (
            23122,
            12,
        ), "The shape of the testing set is not correct."
        assert y_test.shape == (
            23122,
        ), "The shape of labels of testing set is not correct."
        info("Data Shape Assertion test passed.")
    except Exception as e:
        pytest.fail(e)



def test_check_data_leakage():
    """Test if there is any data leakage."""
    try:
        data_frame = ingest_data()
        your_data_cleaner = DataCleaning(data_frame, DataPreprocessStrategy())
        preprocessed_data = your_data_cleaner.handle_data()
        your_data_divider = DataCleaning(preprocessed_data, DataDivideStrategy())
        X_train,X_test,y_train, y_test = your_data_divider.handle_data()
        assert (
            len(X_train.index.intersection(X_test.index)) == 0
        ), "There is data leakage."
        info("Data Leakage test passed.")
    except Exception as e:
        pytest.fail(e)


