import logging
from pathlib import Path

from pandas import read_csv , DataFrame


HOME_DIR = str(Path(__file__).home())
OLIST_CUSTOMERS_DATASET_FPATH = HOME_DIR +  "/dataproduct/MLopsLevelUp1/src/customersatisfaction/Data/olist_customers_dataset.csv"


class IngestData:
    """
    Classe d'ingestion de données qui ingère les données de la source et renvoie un DataFrame.
    """

    def __init__(self) -> None:
        """Initialisation de la classe d'ingestion de données."""
        pass

    def get_data(self) -> DataFrame:
        df = read_csv(OLIST_CUSTOMERS_DATASET_FPATH)
        return df



def ingest_data() -> DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e


