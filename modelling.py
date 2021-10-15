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


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

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


# Validation function
n_folds = 5

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


def train_models(train, y_train):
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
    stacked_averaged_models.fit(train.values, y_train)
    model_xgb.fit(train, y_train)
    model_lgb.fit(train, y_train)
    return stacked_averaged_models, model_xgb, model_lgb


# save/load xgboost model: https://stackoverflow.com/questions/43691380/how-to-save-load-xgboost-model
# save/load lightgbm model:
# https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc
# save/load stacked averaged models:
def save_models():
    model_xgb.save_model("saved_models/model_xgb.model")
    model_lgb.booster_.save_model("saved_models/model_lgb.txt")


def load_models():
    loaded_model_xgb = xgb.XGBRegressor()
    loaded_model_xgb.load_model("saved_models/model_xgb.model")
    loaded_model_lgb = lgb.Booster(model_file="saved_models/model_lgb.txt")
    return loaded_model_xgb, loaded_model_lgb


def test_models(stacked_averaged_models, model_xgb, model_lgb, train, y_train):
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    xgb_train_pred = model_xgb.predict(train)
    lgb_train_pred = model_lgb.predict(train)
    print('RMSLE score on train data [mean squared error regression loss]: ')
    print(rmsle(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15))


def run_predict_models(stacked_averaged_models, model_xgb, model_lgb, test, test_ID):
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    xgb_pred = np.expm1(model_xgb.predict(test))
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('submission.csv', index=False)
