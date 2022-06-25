from collections import defaultdict
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DefaultDict(dict):
    
    def __init__(self, default_item):
        self.default_item = default_item

    # def set_default_item(self, default_item):
    #     self.default_item = default_item
        
    def __missing__(self, key):
        self[key] = self.default_item
        return self.default_item

def create_feature_engineering_pipeline(feature_engineering_step_names=[]):
    feature_engineering_steps = []
    for fe_name in feature_engineering_step_names:
        
        if fe_name == "InstallCumulativeStats":
            feature_engineering_steps.append((fe_name, InstallCumulativeStats()))
        
        if fe_name == "BucketizeTargetEncode":
            feature_engineering_steps.append((fe_name, BucketizeTargetEncode()))
        
        if fe_name == "FeatureBucketize":
            feature_engineering_steps.append((fe_name, FeatureBucketize()))
            
        if fe_name == "TargetEncode":
            feature_engineering_steps.append((fe_name, TargetEncode()))
            
    
    return Pipeline(feature_engineering_steps)


class InstallCumulativeStats(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.granularities = [
            "site",
            "app_site",
            "country_only",
            "adgroup",
            "campaign",
            "network",
            "country",
            "app",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        logging.info("Running InstallCumulativeStats transform")

        X_temp = X.copy()
        for granularity in self.granularities:
            granularity_columns = [f"feature_{granularity}_D-8_installs"]
            for Dx in range(-7, 1):
                granularity_columns.append(f"feature_{granularity}_D{Dx}_installs")
                X_temp[f"feature_{granularity}_D{Dx}_installs_cum_sum"] = X_temp[
                    granularity_columns
                ].sum(axis=1)
                X_temp[f"feature_{granularity}_D{Dx}_installs_cum_mean"] = X_temp[
                    granularity_columns
                ].mean(axis=1)
                X_temp[f"feature_{granularity}_D{Dx}_installs_cum_std"] = X_temp[
                    granularity_columns
                ].std(axis=1)

        return X_temp

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)




# Will not be used directly as it is useless in tree based methods
class FeatureBucketize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bucketizer = KBinsDiscretizer(
            n_bins=20, encode="ordinal", strategy="quantile"
        )

    def fit(self, X, y=None):
        X_temp = X.copy().fillna(-1.0)
        self.bucketizer.fit(X_temp)
        return self

    def transform(self, X):
        bucketized = self.bucketizer.transform(X.fillna(-1.0))
        bucketized_features = [f"{feature}_bucket" for feature in X.columns]
        bucketized = pd.DataFrame(bucketized, columns=bucketized_features)
        return pd.concat([X.reset_index(drop =True), bucketized.reset_index(drop =True)], axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# Will not be used directly as there are no categorical features in the data
class TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self, f: int = 1, k: int = 1):
        self.categories = None
        self.encodings = None
        self.f = f
        self.k = k

    def fit(self, X, y=None):
        self.categories = [col for col in X.columns if "bucket" in col]
        X_temp = X[self.categories].copy()
        X_temp["target"] = y
        self.global_mean = y.mean()
        # self.encodings = defaultdict(lambda: defaultdict(lambda: self.global_mean))
        means_ = DefaultDict(self.global_mean)
        self.encodings = DefaultDict(means_)
        
        for category in self.categories:
            mean = X_temp.groupby(by=category)["target"].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(mean["count"] - self.k) / self.f))
            self.encodings[category] = dict(
                self.global_mean * (1 - smoothing) + mean["mean"] * smoothing
            )
        return self

    def transform(self, X):
        X_temp = X.copy()
        for category in self.categories:
            X_temp[f"{category}_mean_encoding"] = X_temp[category].copy()
            X_temp[f"{category}_mean_encoding"].replace(
                self.encodings[category], inplace=True
            )

            X_temp[f"{category}_mean_encoding"] = X_temp[
                f"{category}_mean_encoding"
            ].astype(float)
        return X_temp

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# Generates categorical features with FeatureBucketize and computes mean encoding
class BucketizeTargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self, f: int = 1, k: int = 1):
        self.f = f
        self.k = k
        self.bucketizer = None
        self.bucket_target_encoder = None

    def fit(self, X, y=None):

        logging.info("Running BucketizeTargetEncode fit")
        
        self.bucketizer = FeatureBucketize()
        self.bucketizer = self.bucketizer.fit(X)

        # End AZ
        X = self.bucketizer.transform(X)

        categories = [col for col in X.columns if "bucket" in col]
        self.bucket_target_encoder = TargetEncode(self.f, self.k)
        self.bucket_target_encoder = self.bucket_target_encoder.fit(X, y)

        return self

    def transform(self, X):

        logging.info("Running BucketizeTargetEncode transform")
        
        X = self.bucketizer.transform(X)
        X = self.bucket_target_encoder.transform(X)
        X = X[
            [
                col
                for col in X.columns
                if ("bucket" not in col or "mean_encoding" in col)
            ]
        ]
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
