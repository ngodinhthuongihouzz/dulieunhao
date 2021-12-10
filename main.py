import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from timeit import default_timer as timer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = timer()

    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points

    # DATA PROCESSING ############
    import data_processing as dp

    train, test = dp.read_multiple_inputs(train_paths=dp.get_all_csv_in_directory('input/train/train.csv'),
                                          test_paths=dp.get_all_csv_in_directory('input/test/test.csv'))

    # train, test = dp.read_input(train_path='input/train.csv', test_path='input/record1.csv')

    train_id, test_id = dp.get_id(train, test)

    train, test = dp.drop_id(train, test)

    # train = dp.delete_outliers(train)  # optional ???

    # check column exists in data-frame:
    # https://stackoverflow.com/questions/24870306/how-to-check-if-a-column-exists-in-pandas
    # We use the numpy function log1p which  applies log(1+x) to all elements of the column
    train["SalePrice"] = np.log1p(train["SalePrice"])  # np.log_e(1+x), e= 2.71828182846

    # Concat train & test data
    all_data, y_train, n_train, n_test = dp.concat_data(train, test)

    # Imputing missing values
    # all_data = dp.imputing_missing_values(all_data)  # optional ???
    # all_data = all_data.drop(['Utilities'], axis=1)  # optional ???
    # all_data = all_data.drop(['Neighborhood'], axis=1)  # optional ???

    # Transforming some numerical variables that are really categorical
    # all_data = dp.transform_numerical_to_categorical_values(all_data)
    all_data = dp.transform_numerical_to_categorical_values_test(all_data)  # customize manually

    # Adding total sq_footage feature
    # all_data = dp.add_more_features(all_data)  # optional ???

    # SKEWED FEATURES
    all_data = dp.box_cox_transform_skewed_features(all_data)

    # Getting new train & test
    all_data, train, test = dp.getting_new_train_test(all_data, n_train)

    print("Data processing time: ", timer() - start)

    # MODELLING ############
    import modelling as mlg

    # Train
    start_train = timer()
    trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb = mlg.train_models(train, y_train)
    # trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb = mlg.train_models_faster(train, y_train)
    print("Train time: ", timer() - start_train)

    # Save trained models
    mlg.save_models(trained_model_xgb, trained_model_lgb, trained_stacked_averaged_models.base_models_,
                    trained_stacked_averaged_models.meta_model_)

    # Load trained models
    # trained_model_xgb, trained_model_lgb, trained_stacked_averaged_models = mlg.load_models()

    # Test
    mlg.test_models(trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb, train, y_train)

    # Predict
    mlg.run_predict_models('output/submission.csv', trained_stacked_averaged_models, trained_model_xgb, trained_model_lgb, test, test_id)

    # Time calculate
    print("Total time: ", timer() - start)

    # Prevent close program
    # k = input("Press any key to exit")

