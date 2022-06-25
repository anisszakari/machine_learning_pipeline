import uuid
import joblib
import _pickle as cPickle
from collections import defaultdict
import itertools
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import logging

from features import create_feature_engineering_pipeline
from constants import (
    RANDOM_SEED,
    DEFAULT_HYPERPARAMETERS,
)


def train_lgbm(
    train_X, train_Y, val_X, val_Y, feature_engineering_steps=None, hyperparameters={}
):
    # Run feature engineering
    feature_engineering_pipeline_id = None
    if feature_engineering_steps is not None:
        logging.info('%s' ,feature_engineering_steps )
        feature_engineering_pipeline_id = str(uuid.uuid4())
        feature_engineering_pipeline = create_feature_engineering_pipeline(
            feature_engineering_steps
        )
        feature_engineering_pipeline = feature_engineering_pipeline.fit(
            train_X, train_Y
        )
        train_X = feature_engineering_pipeline.transform(train_X)
        val_X = feature_engineering_pipeline.transform(val_X)
        
        joblib.dump(
            feature_engineering_pipeline,
            f"feature_pipelines/{feature_engineering_pipeline_id}.pkl",
        )

    # Prepare data
    train_data = lgb.Dataset(train_X, label=train_Y)
    val_data = lgb.Dataset(val_X, label=val_Y)

    # Prepare model parameteres
    model_id = str(uuid.uuid4())
    model_params = {}
    for param in DEFAULT_HYPERPARAMETERS.keys():
        if param in hyperparameters:
            model_params[param] = hyperparameters[param]
        else:
            model_params[param] = DEFAULT_HYPERPARAMETERS[param]

    logging.info('%s', model_params)
    # Train model
    model = lgb.train(
        params=model_params,
        train_set=train_data,
        num_boost_round=10000,
        valid_sets=[train_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=300,
    )
    # Save model to models folder
    joblib.dump(model, f"models/{model_id}.pkl")

    # Compute validation rmse
    val_predictions = model.predict(val_X)
    val_score = mean_squared_error(val_Y, val_predictions, squared=False) if model_params['objective'] == 'regression' else accuracy_score(val_Y, val_predictions)

    # Compute feature importance
    feature_importance = {
        feature_name: float(importance_value)
        for feature_name, importance_value in zip(
            train_X.columns, model.feature_importance()
        )
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    # Return train job config and metrics
    experiment_results = {
        "feature_engineering_pipeline_id": feature_engineering_pipeline_id,
        "feature_engineering_steps": feature_engineering_steps,
        "model_id": model_id,
        "model_parameters": model_params,
        "validation_rmse": val_score,
        "feature_importance": feature_importance,
    }

    return experiment_results


def train_lgbm_kfold(
    X, Y, num_folds=5, feature_engineering_steps=None, hyperparameters={}
):

    # Prepare containers to hold train jobs confgs and metrics
    experiment_results = {
        "fold_results": [],
        "total_results": None,
    }
    val_rmse = 0
    feature_importance = defaultdict(list)

    # Prepare folds
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_num, (train_idx, val_idx) in enumerate(folds.split(X, Y)):

        logging.info(f"[CV] Running fold {fold_num}")

        # Prepare fold train and val data
        train_X = X.iloc[train_idx]
        val_X = X.iloc[val_idx]
        train_Y = Y.iloc[train_idx]
        val_Y = Y.iloc[val_idx]

        # Get fold train job config and metrics
        fold_results = train_lgbm(
            train_X, train_Y, val_X, val_Y, feature_engineering_steps, hyperparameters
        )

        # Append each job result to fold results
        experiment_results["fold_results"].append(fold_results)

        # Compute average val rmse from each fold rmse
        val_rmse += fold_results["validation_rmse"] ** 2
        # Compute average feature importance from each fold feature importance
        for feature, importance in fold_results["feature_importance"].items():
            feature_importance[feature].append(importance)

    # Compute average val rmse from each fold rmse
    val_rmse = (val_rmse / num_folds) ** 0.5
    # Compute average feature importance from each fold feature importance
    for feature in feature_importance:
        feature_importance[feature] = sum(feature_importance[feature]) / len(
            feature_importance[feature]
        )
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    # Add average rmse and feature importance to run results
    experiment_results["total_results"] = {
        "validation_rmse": val_rmse,
        "feature_importance": feature_importance,
    }
    return experiment_results


def train_lgbm_grid_search_kfold(
    X, Y, num_folds=5, feature_engineering_steps=None, hyperparameters={}
):
    # Make sure we have a list for every hyperparam we want to tune
    for hyperparameter in hyperparameters:
        if type(hyperparameters[hyperparameter]) != list:
            hyperparameters[hyperparameter] = [hyperparameters[hyperparameter]]

    # preparee container for every grid search run results
    grid_search_results = {"all_grid_search_runs": [], "best_grid_search_run": None}

    # Iterate over grd search space
    for grid_search_instance in itertools.product(*hyperparameters.values()):
        grid_search_hyperparameters = dict(
            zip(hyperparameters.keys(), grid_search_instance)
        )

        logging.info(f"[GS] grid search params: {grid_search_hyperparameters}")

        # Get grid search CV runs results
        experiment_results = train_lgbm_kfold(
            X,
            Y,
            num_folds=num_folds,
            feature_engineering_steps=feature_engineering_steps,
            hyperparameters=grid_search_hyperparameters,
        )
        # Append grid search run to grid search runs
        grid_search_results["all_grid_search_runs"].append(experiment_results)

        # Update best grid search run
        if grid_search_results["best_grid_search_run"] is None:
            grid_search_results["best_grid_search_run"] = experiment_results
        elif (
            experiment_results["total_results"]["validation_rmse"]
            < grid_search_results["best_grid_search_run"]["total_results"][
                "validation_rmse"
            ]
        ):
            grid_search_results["best_grid_search_run"] = experiment_results

    return grid_search_results
