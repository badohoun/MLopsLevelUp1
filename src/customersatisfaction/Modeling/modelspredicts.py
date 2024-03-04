import mlflow 
import mlflow.sklearn 



def load_artifacts():
    run_id = mlflow.list_run_infos("3")[0].run_id
    model = mlflow.sklearn.load_model("runs:{}/model".format(run_id))
    return model