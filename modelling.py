import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)


#
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#
#     # we define clones of the original models to fit the data in
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)
#
#         return self
#
#     # Now we do the predictions for cloned models and average them
#     def predict(self, X):
#         predictions = np.column_stack([model.predict(X) for model in self.models_])
#         return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):
        # base models: ENet, GBoost, KRR
        self.base_models = base_models
        # meta models: lasso
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        # [DEBUG]
        print("[DEBUG] X: ", X.shape, ", y: ", y.shape)

        self.base_models_ = [list() for x in self.base_models]  # convert tuple base_models to list base_models_
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])  # todo: check memory problem here:
                # numpy.core._exceptions.MemoryError: Unable to allocate 30.2 = float64 (8 byte * 63671 * 63671 / (
                # 1024*1024*1024)) GiB for an array with shape (63671, 63671) and data type float64
                # https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type [TEST]
                print("[DEBUG] ", i, ":", train_index, ":",  train_index.size, ":", holdout_index, ":", holdout_index.size)
                import sys
                print("[DEBUG] instance SZ=", sys.getsizeof(instance))
                print("[DEBUG] X SZ=", sys.getsizeof(X[train_index]), ", y SZ=", sys.getsizeof(y[train_index]))
                import os, psutil
                process = psutil.Process(os.getpid())
                # print("MemoryInfo: ", process.memory_info().rss / (1024 * 1024), "MB")  # in MB
                # print("MemoryInfo pagefile: ", process.memory_info().pagefile)
                # print("MemoryInfo peak_pagefile: ", process.memory_info().peak_pagefile)
                # print("MemoryInfo paged_pool: ", process.memory_info().paged_pool)

                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred  # todo: break here to get output

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    def load(self, base_models_, meta_model_):
        self.base_models_ = base_models_
        self.meta_model_ = meta_model_

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def rmsle_cv(model, train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    # rmse = cross_val_score(model, train.values, y_train, scoring="mean_squared_error", cv=5)
    return (rmse)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def accuracy_loss_percentage(y, y_pred):
    return np.mean((abs(y_pred - y) / y) * 100)


def accuracy_correct_percentage(y, y_pred):
    return 100 - np.mean((abs(y_pred - y) / y) * 100)


# Validation function
n_folds = 5

# no n_jobs?
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# no n_jobs?
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# no n_jobs?
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# no n_jobs?
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)

# check to add to increase speed n_jobs = threading.active_count(), nthread = threading.active_count()
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

# n_jobs = threading.active_count()
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

import multiprocessing


def train_models_faster(train, y_train):
    if multiprocessing.cpu_count() < 3:
        raise Exception("[RE-AVM Error] CPU limited (< 3)")

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    stacked_avg_train_process = multiprocessing.Process(target=train_stacked_averaged_model,
                                                        args=(train, y_train, return_dict,))
    processes.append(stacked_avg_train_process)

    xgb_train_process = multiprocessing.Process(target=train_xgb_model, args=(train, y_train, return_dict,))
    processes.append(xgb_train_process)

    lgb_train_process = multiprocessing.Process(target=train_lgb_model, args=(train, y_train, return_dict,))
    processes.append(lgb_train_process)

    stacked_avg_train_process.start()
    xgb_train_process.start()
    lgb_train_process.start()

    for process in processes:
        process.join()
    return return_dict["trained_stacked_averaged_models"], return_dict["trained_model_xgb"], return_dict[
        "trained_model_lgb"]


def train_stacked_averaged_model(train, y_train, return_dict):
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    stacked_averaged_models.fit(train.values, y_train)
    return_dict["trained_stacked_averaged_models"] = stacked_averaged_models


def train_xgb_model(train, y_train, return_dict):
    model_xgb.fit(train, y_train)
    return_dict["trained_model_xgb"] = model_xgb


def train_lgb_model(train, y_train, return_dict):
    model_lgb.fit(train, y_train)
    return_dict["trained_model_lgb"] = model_lgb


def train_models(train, y_train):
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    stacked_averaged_models.fit(train.values, y_train)
    model_xgb.fit(train, y_train)
    model_lgb.fit(train, y_train)
    return stacked_averaged_models, model_xgb, model_lgb


# save/load xgboost model: https://stackoverflow.com/questions/43691380/how-to-save-load-xgboost-model
# save/load lightgbm model:
# https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc
# save stacked averaged models:
# https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
# load stacked averaged models:
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
def save_models(trained_model_xgb, trained_model_lgb, trained_base_models_, trained_meta_model_):
    trained_model_xgb.save_model("output/saved_models/trained_model_xgb.model")
    trained_model_lgb.booster_.save_model("output/saved_models/trained_model_lgb.txt")
    import joblib
    joblib.dump(trained_base_models_, "output/saved_models/trained_base_models_.pkl")
    joblib.dump(trained_meta_model_, "output/saved_models/trained_meta_model_.pkl")


def load_models():
    loaded_trained_model_xgb = xgb.XGBRegressor()
    loaded_trained_model_xgb.load_model("output/saved_models/trained_model_xgb.model")
    loaded_trained_model_lgb = lgb.Booster(model_file="output/saved_models/trained_model_lgb.txt")
    import joblib
    trained_base_models_ = joblib.load("output/saved_models/trained_base_models_.pkl")
    trained_meta_model_ = joblib.load("output/saved_models/trained_meta_model_.pkl")
    loaded_trained_stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    loaded_trained_stacked_averaged_models.load(trained_base_models_, trained_meta_model_)
    return loaded_trained_model_xgb, loaded_trained_model_lgb, loaded_trained_stacked_averaged_models


def test_models(trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb, train, y_train):
    stacked_train_pred = trained_stacked_averaged_models.predict(train.values)
    xgb_train_pred = trained_model_xgb.predict(train)
    lgb_train_pred = trained_model_lgb.predict(train)
    print('RMSLE score on train data [mean squared error regression loss]: ')
    print(rmsle(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15))

    print('ACCURACY score on train data [mean accuracy loss percentage]: ')
    print(accuracy_loss_percentage(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15), "%")

    print('ACCURACY score on train data [mean accuracy correction percentage]: ')
    print(accuracy_correct_percentage(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15), "%")


def run_predict_models(out_path, trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb, test, test_ID):
    stacked_pred = np.expm1(trained_stacked_averaged_models.predict(test.values))
    xgb_pred = np.expm1(trained_model_xgb.predict(test))
    lgb_pred = np.expm1(trained_model_lgb.predict(test.values))
    ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv(out_path, index=False)
    return sub['SalePrice']

