import uuid
import yaml
import json
import os
import pandas as pd
from datetime import datetime
import warnings
import logging
import re

from model import train_lgbm_grid_search_kfold

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    warnings.filterwarnings("ignore")

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("feature_pipelines", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    experiment_id = str(uuid.uuid4())

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            exit()

    start_datetime = str(datetime.now())

    logging.info("Loading train data")
    train_data_path = config["data"]["train_data"]
    train_data = pd.read_csv(train_data_path, sep=config["data"]["separator"]).rename(columns = {config["data"]["target"] : "target"})
    train_data = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    print(train_data.head())
    trainX = train_data[[col for col in train_data.columns if col != "target"]]
    trainY = train_data["target"]
    logging.info("Finished loading train data")

    feature_engineering_steps = config["feature_engineering"]
    num_folds = config["num_folds"]
    hyperparameters = config["hyperparameters"]
    result = train_lgbm_grid_search_kfold(
        trainX, trainY, num_folds, feature_engineering_steps, hyperparameters
    )

    end_datetime = str(datetime.now())

    result["start_datetime"] = start_datetime
    result["end_datetime"] = end_datetime
    result["train_data_path"] = train_data_path

    with open(f"experiments/{experiment_id}.json", "w") as fp:
        json.dump(result, fp, indent=4)
