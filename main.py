import data_processing as dp
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points

# DATA PROCESSING ############
def data_processing():


train, test = dp.read_input(train_path='input/train.csv', test_path='input/test.csv')

train_id, test_id = dp.get_id(train, test)

train, test = dp.drop_id(train, test)

train = dp.delete_outliers(train)  # optional optimizations ???

# We use the numpy function log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])  # np.log_e(1+x), e= 2.71828182846

# Concat train & test data
all_data, y_train, n_train, n_test = dp.concat_data(train, test)

# Imputing missing values
all_data = dp.imputing_missing_values(all_data)

# Transforming some numerical variables that are really categorical
all_data = dp.transform_numerical_to_categorical_values(all_data)

# Adding total sq_footage feature
all_data = dp.add_more_features(all_data)

# SKEWED FEATURES
all_data = dp.box_cox_transform_skewed_features(all_data)

# Getting new train & test
all_data, train, test = dp.getting_new_train_test(all_data, n_train)

# MODELLING ############
import modelling as mlg

# Train
stacked_averaged_models, model_xgb, model_lgb = mlg.train_models(train, y_train)

# Test
mlg.test_models(stacked_averaged_models, model_xgb, model_lgb, train, y_train)

# Predict
mlg.run_predict_models(stacked_averaged_models, model_xgb, model_lgb, test, test_id)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
