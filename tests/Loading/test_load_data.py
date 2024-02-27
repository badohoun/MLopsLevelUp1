from pathlib import Path
from logging import info 
from io import StringIO
import pytest
from pandas import read_csv 
from customersatisfaction.Loading.load_data import  ingest_data





# Remplacez ceci par le chemin réel vers votre fichier CSV

HOME_DIR = str(Path(__file__).home())
OLIST_CUSTOMERS_DATASET_FPATH = HOME_DIR +  "/dataproduct/MLopsLevelUp1/src/customersatisfaction/Data/olist_customers_dataset.csv"


@pytest.fixture
def sample_data():
    # Charger un échantillon de données pour les tests
    data = """
order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date,payment_sequential,payment_type,payment_installments,payment_value,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm,product_category_name_english,review_score,review_comment_message
e481f51cbdc54678b7cc49136f2d6af7,9ef432eb6251297304e76186b10a928d,delivered,2017-10-02 10:56:33,2017-10-02 11:07:15,2017-10-04 19:55:00,2017-10-10 21:25:13,2017-10-18 00:00:00,1,credit_card,1,18.12,7c396fd4830fd04220f754e42b4e5bff,3149,sao paulo,SP,1,87285b34884572647811a353c7ac498a,3504c0cb71d7fa48d967e0e4c94d59d9,2017-10-06 11:07:15,29.99,8.72,utilidades_domesticas,40.0,268.0,4.0,500.0,19.0,8.0,13.0,housewares,4,"Não testei o produto ainda, mas ele veio correto e em boas condições. Apenas a caixa que veio bem amassada e danificada, o que ficará chato, pois se trata de um presente."
e481f51cbdc54678b7cc49136f2d6af7,9ef432eb6251297304e76186b10a928d,delivered,2017-10-02 10:56:33,2017-10-02 11:07:15,2017-10-04 19:55:00,2017-10-10 21:25:13,2017-10-18 00:00:00,3,voucher,1,2.0,7c396fd4830fd04220f754e42b4e5bff,3149,sao paulo,SP,1,87285b34884572647811a353c7ac498a,3504c0cb71d7fa48d967e0e4c94d59d9,2017-10-06 11:07:15,29.99,8.72,utilidades_domesticas,40.0,268.0,4.0,500.0,19.0,8.0,13.0,housewares,4,"Não testei o produto ainda, mas ele veio correto e em boas condições. Apenas a caixa que veio bem amassada e danificada, o que ficará chato, pois se trata de um presente."
    """
    df = read_csv(StringIO(data))
    return df.head(2)

def test_ingest_data_loads_data_correctly(sample_data):
    # Teste si la classe IngestData charge correctement les données
    raw_df = ingest_data().head(2)
    assert raw_df.equals(sample_data)


@pytest.mark.parametrize(
    "column_name, expected_dtype",
    [
        ("order_purchase_timestamp", "object"),
        ("payment_value", "float64"),
        ("review_score", "int64")
        # Ajoutez d'autres colonnes ici avec leurs types de données attendus
    ],
)
def test_data_types(column_name, expected_dtype):
    # Teste si les types de données des colonnes sont corrects
    sample_data = ingest_data()
    assert sample_data[column_name].dtype == expected_dtype

