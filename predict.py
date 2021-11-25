import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from timeit import default_timer as timer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = timer()

    # DATA PROCESSING ############
    import data_processing as dp

    record = dp.read_single_predict_input(predict_path='input/predict/record1.csv')
    print("***Input: \n", record.to_string())
    record_id = record['Id']
    record.drop("Id", axis=1, inplace=True)
    # Imputing missing values
    # record = dp.imputing_missing_values(record)
    # record = record.drop(['Utilities'], axis=1)
    # Transforming some numerical variables that are really categorical
    # record = dp.load_transform_numerical_to_categorical_values(record)
    record = dp.record_load_transform_numerical_to_categorical_values(record)
    # Adding total sq_footage feature
    # record = dp.add_more_features(record)
    # SKEWED FEATURES
    record = dp.box_cox_transform_skewed_features_loaded(record)

    print("type: ", type(record))
    # Getting new record

    print(record.shape)

    record = pd.get_dummies(record)

    # PREDICTING ############
    import modelling as mlg

    # Load trained models
    trained_model_xgb, trained_model_lgb, trained_stacked_averaged_models = mlg.load_models()

    # Predict
    out = mlg.run_predict_models('output/predict/submission.csv', trained_stacked_averaged_models, trained_model_xgb,
                                 trained_model_lgb, record, record_id)

    print("***Output: \n", out.to_string())

    # Time calculate
    print("Predict time: ", timer() - start)

    # Prevent close program
    # k = input("Press any key to exit")

