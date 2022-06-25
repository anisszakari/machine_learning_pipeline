import numpy as np
import yaml
import os
import pandas as pd
import json
import joblib
import warnings
import logging

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    warnings.filterwarnings("ignore")

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            exit()

    logging.info("Loading test data")

    test_data_path = config["data"]["test_data"]
    pred_data_path = config["data"]["prediction_path"]

    test_data = pd.read_csv(test_data_path, sep=config["data"]["separator"]).rename(columns = {config["data"]["target"] : "target"})
    testX = test_data[[col for col in test_data.columns if col != "target"]]

    logging.info("Finished loading test data")

    # Get best experiment
    logging.info("Selecting best experiment we have")
    best_rmse = None
    best_feature_pipeline_ids = None
    best_model_ids = None
    for experiment_path in os.listdir("experiments/"):
        if experiment_path.endswith('.json') :
            with open(f"experiments/{experiment_path}") as fp:
                experiment = json.load(fp)
                experiment = experiment["best_grid_search_run"]
                if (
                    best_rmse is None
                    or best_rmse > experiment["total_results"]["validation_rmse"]
                ):
                    best_feature_pipeline_ids = [
                        x["feature_engineering_pipeline_id"]
                        for x in experiment["fold_results"]
                    ]
                    best_model_ids = [x["model_id"] for x in experiment["fold_results"]]

    # Run predictions
    logging.info(
        "Best KFold feature engineering pipelines: %s", best_feature_pipeline_ids
    )
    logging.info("Best KFold models: %s", best_model_ids)

    num_folds = len(best_model_ids)
    predictions = np.zeros(testX.shape[0])
    for feature_pipeline_id, model_id in zip(best_feature_pipeline_ids, best_model_ids):
        logging.info("Running Predictions")
        if feature_pipeline_id is not None :
            feature_pipeline = joblib.load(f"feature_pipelines/{feature_pipeline_id}.pkl")
            print('s%',feature_pipeline)
            print(testX.head())
            processedtestX = feature_pipeline.transform(testX)
            
        
        model = joblib.load(f"models/{model_id}.pkl")
        predictions += model.predict(processedtestX) / num_folds

    logging.info("Saving Predictions")
    test_data["predictions"] = predictions
    test_data.to_csv(pred_data_path, index=False)
