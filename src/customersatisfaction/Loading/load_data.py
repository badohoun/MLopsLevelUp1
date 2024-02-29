import logging
from pathlib import Path

from pandas import read_csv, DataFrame


THIS_DIR = Path(__file__).parent

OLIST_CUSTOMERS_CSV_FPATH = THIS_DIR /  "../Data/olist_customers_dataset.csv"

class IngestData:
    """
    Classe d'ingestion de données qui ingère les données de la source et renvoie un DataFrame.
    """

    def __init__(self) -> None:
        """Initialisation de la classe d'ingestion de données."""
        pass

    def get_data(self) -> DataFrame:
        df = read_csv(OLIST_CUSTOMERS_CSV_FPATH)
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

