from logging import info , error
from abc import ABC, abstractmethod

from numpy import  ndarray , sqrt 
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            info("Entered the calculate_score method of the MSE class")
            mse = mean_squared_error(y_true, y_pred)
            info("The mean squared error value is: " + str(mse))
            return mse
        except Exception as e:
            error(
                "Exception occurred in calculate_score method of the MSE class. Exception message:  "
                + str(e)
            )
            raise e


class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            info("Entered the calculate_score method of the R2Score class")
            r2 = r2_score(y_true, y_pred)
            info("The r2 score value is: " + str(r2))
            return r2
        except Exception as e:
            error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
                + str(e)
            )
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            info("Entered the calculate_score method of the RMSE class")
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
                + str(e)
            )
            raise e