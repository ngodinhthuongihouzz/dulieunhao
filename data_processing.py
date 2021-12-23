import csv

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm, skew  # for some statistics


def read_input(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def read_multiple_inputs(train_paths, test_paths):
    train = pd.DataFrame()
    for train_path in train_paths:
        train_i = pd.read_csv(train_path)
        train = train.append(train_i, ignore_index=True)

    test = pd.DataFrame()
    for test_path in test_paths:
        test_i = pd.read_csv(test_path)
        test = test.append(test_i, ignore_index=True)
    return train, test


def read_single_predict_input(predict_path):
    return pd.read_csv(predict_path)


def get_all_csv_in_directory(dir_path):
    import glob
    return glob.glob(dir_path)


def get_id(train, test):
    # Save the 'Id' column
    train_id = train['Id']
    test_id = test['Id']
    return train_id, test_id


def vn_get_id(train, test):
    # Save the 'ID' column
    train_id = train['ID']
    test_id = test['ID']
    return train_id, test_id


def drop_id(train, test):
    # Drop the  'Id' column since it's unnecessary for  the prediction process.
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test


def vn_drop_id(train, test):
    # Drop the  'ID' column since it's unnecessary for  the prediction process.
    train.drop("ID", axis=1, inplace=True)
    test.drop("ID", axis=1, inplace=True)
    return train, test


# Optional for each criteria to improve over-fitting, can be omitted
def delete_outliers(train):
    # Deleting outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    return train


def concat_data(train, test):
    # Concat train & test data
    n_train = train.shape[0]
    n_test = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    return all_data, y_train, n_train, n_test


def vn_concat_data(train, test):
    # Concat train & test data
    n_train = train.shape[0]
    n_test = test.shape[0]
    y_train = train.Price.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['Price'], axis=1, inplace=True)
    return all_data, y_train, n_train, n_test


def show_missing_values(all_data):
    # Missing data
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data.head(20))


def imputing_missing_values(all_data):
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    return all_data


def vn_imputing_missing_values(all_data):
    # TEST all datatype of columns
    # print("INSPECT ALL COLUMNS'S DATATYPE [BEFORE IMPUTE]")
    # for col in all_data.columns:
    #     print("type of ", col, ": ", all_data[col].dtypes)

    # todo refilling values here for categorical features which has values is number ("None" -> 0.0)
    all_data["FormRE"] = all_data["FormRE"].fillna(0)
    all_data["Province"] = all_data["Province"].fillna(0)
    all_data["FloorNum"] = all_data["FloorNum"].fillna(0)
    all_data["Area"] = all_data["Area"].fillna(0)
    all_data["Height"] = all_data["Height"].fillna(0)
    all_data["Width"] = all_data["Width"].fillna(0)
    all_data["UsableArea"] = all_data["UsableArea"].fillna(0)
    all_data["FrontLength"] = all_data["FrontLength"].fillna(0)
    all_data["BackSideLength"] = all_data["BackSideLength"].fillna(0)
    all_data["Direction"] = all_data["Direction"].fillna(0)
    all_data["BalconyDirection"] = all_data["BalconyDirection"].fillna(0)
    all_data["Corner"] = all_data["Corner"].fillna("None")
    all_data["RoadInFront"] = all_data["RoadInFront"].fillna(0)
    all_data["Juridical"] = all_data["Juridical"].fillna(0)
    all_data["NumOfBed"] = all_data["NumOfBed"].fillna(0)
    all_data["NumOfFloor"] = all_data["NumOfFloor"].fillna(0)
    all_data["NumOfToilet"] = all_data["NumOfToilet"].fillna(0)
    all_data["ConstructionYear"] = all_data["ConstructionYear"].fillna(0)
    all_data["IsOwner"] = all_data["IsOwner"].fillna("None")
    all_data["Furniture"] = all_data["Furniture"].fillna("None")
    all_data["Terrace"] = all_data["Terrace"].fillna("None")
    all_data["CarParking"] = all_data["CarParking"].fillna("None")
    all_data["DinningRoom"] = all_data["DinningRoom"].fillna("None")
    all_data["Kitchen"] = all_data["Kitchen"].fillna("None")
    all_data["AirCond"] = all_data["AirCond"].fillna("None")
    all_data["ADSL"] = all_data["ADSL"].fillna("None")
    all_data["WashingMachine"] = all_data["WashingMachine"].fillna("None")
    all_data["Balcony"] = all_data["Balcony"].fillna("None")
    all_data["Fridge"] = all_data["Fridge"].fillna("None")
    all_data["Wifi"] = all_data["Wifi"].fillna("None")
    all_data["Pool"] = all_data["Pool"].fillna("None")
    all_data["Basement"] = all_data["Basement"].fillna("None")
    all_data["Park"] = all_data["Park"].fillna("None")
    all_data["SuperMarket"] = all_data["SuperMarket"].fillna("None")
    all_data["Clinics"] = all_data["Clinics"].fillna("None")
    all_data["Sea"] = all_data["Sea"].fillna("None")
    all_data["Hospital"] = all_data["Hospital"].fillna("None")
    all_data["Church"] = all_data["Church"].fillna("None")
    all_data["BusStation"] = all_data["BusStation"].fillna("None")
    all_data["School"] = all_data["School"].fillna("None")
    all_data["Temple"] = all_data["Temple"].fillna("None")
    all_data["Airport"] = all_data["Airport"].fillna("None")
    all_data["Preschool"] = all_data["Preschool"].fillna("None")
    all_data["Characteristics"] = all_data["Characteristics"].fillna(0)
    # all_data["PricePSM"] = all_data["PricePSM"].fillna(0)

    # TEST all datatype of columns
    # print("INSPECT ALL COLUMNS'S DATATYPE [AFTER IMPUTE]")
    # for col in all_data.columns:
    #     print("type of ", col, ": ", all_data[col].dtypes)

    return all_data


def check_remaining_missing_values(all_data):
    # Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data.head(n=100))  # default for first 100 features
    # show figure for missing values
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()


def check_remaining_missing_values_and_remove(all_data):
    # Check remaining missing values if any
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data.head(n=100))  # default for first 100 features
    # show figure for missing values
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()
    data = drop_data_and_save(all_data, all_data_na, 'input/crawled/train20211207_110259[Dropped].csv')
    return data


def remove_unset_rows(data):
    return data


def fill_unset_rows(data):
    area = data["Area"].tolist()  # Diện Tích
    height = data["Height"].tolist()  # Chiều Dài
    width = data["Width"].tolist()  # Chiều Ngang
    import math
    # print("len area: ", len(area))
    # print("len height: ", len(height))
    # print("area[0].isnan(): ", math.isnan(area[0]))
    # print("height[rec].isnan(): ", math.isnan(height[0]))
    for rec in range(len(area)):
        if math.isnan(area[rec]) and math.isnan(height[rec]) is False and math.isnan(width[rec]) is False:
            data["Area"][rec] = height[rec] * width[rec]
            # data["Area"].set_value()
            # data.set_value(rec, "Area", height[rec] * width[rec], takeable=False)
            # data.

    data.to_csv('input/crawled/train20211207_110259[Input].csv', index=False, encoding='utf-8-sig')
    return data


# Transforming some numerical variables that are really categorical
def convert_num_to_string_values(all_data):
    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    # Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    # Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    return all_data


def record_convert_num_to_string_values(all_data):
    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    # Changing OverallCond into a categorical variable
    # all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    # Year and month sold are transformed into categorical features.
    # all_data['YrSold'] = all_data['YrSold'].astype(str)
    # all_data['MoSold'] = all_data['MoSold'].astype(str)
    return all_data


# todo: this function only apply for columns which has categorical features,
#  and it was presented by numbers
def vn_convert_num_to_string_values(all_data):
    # TEST all datatype of columns
    # print("INSPECT ALL COLUMNS'S DATATYPE [BEFORE CONVERT TO STRING]")
    # for col in all_data.columns:
    #     print("type of ", col, ": ", all_data[col].dtypes)

    # apply string here for number
    all_data['FormRE'] = all_data['FormRE'].apply(str)
    all_data['Province'] = all_data['Province'].apply(str)
    all_data['Direction'] = all_data['Direction'].apply(str)
    all_data['BalconyDirection'] = all_data['BalconyDirection'].apply(str)
    all_data['Juridical'] = all_data['Juridical'].apply(str)
    all_data['Characteristics'] = all_data['Characteristics'].apply(str)

    # all_data['BackSideLength'] = all_data['BackSideLength'].apply(str)
    # all_data['RoadInFront'] = all_data['RoadInFront'].apply(str)
    # all_data['NumOfBed'] = all_data['NumOfBed'].apply(str)
    # all_data['NumOfFloor'] = all_data['NumOfFloor'].apply(str)
    # all_data['NumOfToilet'] = all_data['NumOfToilet'].apply(str)
    # all_data['ConstructionYear'] = all_data['ConstructionYear'].apply(str)
    # all_data['PricePSM'] = all_data['PricePSM'].apply(str)

    # # Changing OverallCond into a categorical variable
    # all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    # # Year and month sold are transformed into categorical features.
    # all_data['YrSold'] = all_data['YrSold'].astype(str)
    # all_data['MoSold'] = all_data['MoSold'].astype(str)

    # TEST all datatype of columns
    # print("INSPECT ALL COLUMNS'S DATATYPE [AFTER CONVERT TO STRING]")
    # for col in all_data.columns:
    #     print("type of ", col, ": ", all_data[col].dtypes)

    return all_data


# SAVE LabelEncoder
# https://stackoverflow.com/questions/28656736/using-scikits-labelencoder-correctly-across-multiple-programs
def label_encoding(all_data):
    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold'
            )
    # process columns, apply LabelEncoder to categorical features
    # convert categorical features to number
    lbl_dict = dict()
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))  # get all type of categorical features
        # print("[test]type of values: ", type(all_data[c].values))
        # print("[test] type of list values: ", type(list(all_data[c].values)))
        lbl_dict[c] = lbl
        all_data[c] = lbl.transform(list(all_data[c].values))  # convert categorical features to number

    # SAVE
    np.save('output/saved_models/label_encoder.npy', lbl_dict)
    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def label_encoding_test(all_data):
    # check type here
    # print("type of columns LotFrontage: ", type(all_data['LotFrontage'][0]))
    # print("values: ", all_data['LotFrontage'].dtypes)
    # print("type of columns Neighborhood: ", type(all_data['Neighborhood'][0]))
    # print("values: ", all_data['Neighborhood'].dtypes)

    # print("INSPECT ALL COLUMNS'S DATATYPE")
    # for col in all_data.columns:
    #     print("type of ", col, ": ", all_data[col].dtypes)

    from sklearn.preprocessing import LabelEncoder
    cols = ('MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
            'LotConfig', 'LandSlope', 'Neighborhood'
            )

    # process columns, apply LabelEncoder to categorical features
    # convert categorical features to number
    lbl_dict = dict()
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))  # get all type of categorical features
        # print("[test]type of values: ", type(all_data[c].values))
        # print("[test] type of list values: ", type(list(all_data[c].values)))
        lbl_dict[c] = lbl
        all_data[c] = lbl.transform(list(all_data[c].values))  # convert categorical features to number

    # SAVE
    np.save('output/saved_models/label_encoder.npy', lbl_dict)
    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


# todo: ignore dummy data by labeling all categorical features to number
def vn_label_encoding(all_data):
    print("INSPECT ALL COLUMNS'S DATATYPE [LABEL ENCODING]")
    for col in all_data.columns:
        print("type of ", col, ": ", all_data[col].dtypes)

    from sklearn.preprocessing import LabelEncoder
    # todo: add all categorical features here
    cols = ('FormRE', 'Province', 'Direction', 'BalconyDirection', 'Corner',
            'Juridical', 'IsOwner', 'Furniture', 'Terrace', 'CarParking', 'DinningRoom',
            'Kitchen', 'AirCond', 'ADSL', 'WashingMachine', 'Balcony', 'Fridge',
            'Wifi', 'Pool', 'Basement', 'Park', 'SuperMarket', 'Clinics', 'Sea',
            'Hospital', 'Church', 'BusStation', 'School', 'Temple', 'Airport',
            'Preschool', 'Characteristics')  # không chừa một cột nào còn
    # Width is categorical or number (categorical), todo add 'Width'
    # giá trị categorial -> warning "UserWarning: Usage of np.ndarray subset (sliced data) is not recommended due
    # to it will double the peak memory cost in LightGBM. _log_warning("Usage of np.ndarray subset (sliced data) is
    # not recommended " process columns, apply LabelEncoder to categorical features convert categorical features to
    # number
    lbl_dict = dict()
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))  # get all type of categorical features
        # print("[test]type of values: ", type(all_data[c].values))
        # print("[test] type of list values: ", type(list(all_data[c].values)))

        lbl_dict[c] = lbl
        all_data[c] = lbl.transform(list(all_data[c].values))  # convert categorical features to number

    # SAVE
    np.save('output/saved_models/label_encoder.npy', lbl_dict)
    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def label_encoding_loaded(all_data):
    ori_encoder_dict = np.load('output/saved_models/label_encoder.npy', allow_pickle=True)
    encoder_dict = ori_encoder_dict.tolist()
    for key in encoder_dict.keys():
        lbl = encoder_dict[key]
        all_data[key] = lbl.transform(list(all_data[key].values))  # convert categorical features to number

    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def vn_label_encoding_loaded(all_data):
    ori_encoder_dict = np.load('output/saved_models/label_encoder.npy', allow_pickle=True)
    encoder_dict = ori_encoder_dict.tolist()
    for key in encoder_dict.keys():
        lbl = encoder_dict[key]
        all_data[key] = lbl.transform(list(all_data[key].values))  # convert categorical features to number

    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def transform_numerical_to_categorical_values(all_data):
    all_data = convert_num_to_string_values(all_data)
    all_data = label_encoding(all_data)
    return all_data


def transform_numerical_to_categorical_values_test(all_data):
    all_data = record_convert_num_to_string_values(all_data)
    all_data = label_encoding_test(all_data)
    return all_data


def vn_transform_numerical_to_categorical_values(all_data):
    all_data = vn_convert_num_to_string_values(all_data)
    all_data = vn_label_encoding(all_data)
    return all_data


def load_transform_numerical_to_categorical_values(all_data):
    all_data = convert_num_to_string_values(all_data)
    all_data = label_encoding_loaded(all_data)
    return all_data


def record_load_transform_numerical_to_categorical_values(all_data):
    all_data = record_convert_num_to_string_values(all_data)
    all_data = label_encoding_loaded(all_data)
    return all_data


def vn_record_load_transform_numerical_to_categorical_values(all_data):
    all_data = vn_convert_num_to_string_values(all_data)
    all_data = vn_label_encoding_loaded(all_data)
    return all_data


def add_more_features(all_data):
    # Since area related features are very important to determine house prices, we add one more feature
    # which is the total area of basement, first and second floor areas of each house
    # Adding total sqfootage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    return all_data


def skewed_features_func(all_data):
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    # print(skewness.head(10))
    return skewness


def box_cox_transform_skewed_features(all_data):
    skewness = skewed_features_func(all_data)
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    # print("[check-before]:", all_data["LotArea"])
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # all_data[skewed_features] = np.log1p(all_data[skewed_features]) # like this when lam = 0

    # SAVE
    np.save('output/saved_models/skewed_features.npy', skewed_features, allow_pickle=True)

    return all_data


def box_cox_transform_skewed_features_loaded(record):
    from scipy.special import boxcox1p
    # LOAD
    skewed_features = np.load('output/saved_models/skewed_features.npy', allow_pickle=True)
    lam = 0.15
    # print("[check-before]:", all_data["LotArea"])
    for feat in skewed_features:
        # all_data[feat] += 1
        record[feat] = boxcox1p(record[feat], lam)

    return record


# dummy data, create all columns of categorical which are available
# example: MSZoning : RL/RH -> dummy will create 2 columns: MSZoning_RL , MSZoning_RH
def getting_new_train_test(all_data, n_train):
    all_data = pd.get_dummies(all_data)
    print("Size after get dummies:", all_data.shape)
    train = all_data[:n_train]
    test = all_data[n_train:]
    return all_data, train, test


def convert_crawled_to_input_data(crawled_paths, out_train_file_path, out_test_file_path):
    from timeit import default_timer as timer
    start = timer()
    print("start convert_crawled_to_input_data ...")

    crawled_datas = pd.DataFrame()
    for crawled_path in crawled_paths:
        crawled_data = pd.read_csv(crawled_path)
        crawled_datas = crawled_datas.append(crawled_data, ignore_index=True)

    # TEST all datatype of columns
    # print("INSPECT ALL COLUMNS'S DATATYPE")
    # for col in crawled_data.columns:
    #     print("type of ", col, ": ", crawled_datas[col].dtypes)

    # todo: get each row of data to list
    form_re = crawled_datas["FormRE"].tolist()  # Loại Hình BĐS
    province = crawled_datas["Province"].tolist()  # Tỉnh Thành
    floor_num = crawled_datas["FloorNum"].tolist()  # Tầng Số
    area = crawled_datas["Area"].tolist()  # Diện Tích
    height = crawled_datas["Height"].tolist()  # Chiều Dài
    width = crawled_datas["Width"].tolist()  # Chiều Rộng
    usable_area = crawled_datas["UsableArea"].tolist()  # DTSD
    front_length = crawled_datas["FrontLength"].tolist()  # Mặt Tiền
    back_side_length = crawled_datas["BackSideLength"].tolist()  # Mặt Hậu
    direction = crawled_datas["Direction"].tolist()  # Hướng
    balcony_direction = crawled_datas["BalconyDirection"].tolist()  # Hướng Ban Công
    corner = crawled_datas["Corner"].tolist()  # Căn Góc
    road_in_front = crawled_datas["RoadInFront"].tolist()  # Đường Trước Nhà
    juridical = crawled_datas["Juridical"].tolist()  # Pháp Lý
    num_of_bed = crawled_datas["NumOfBed"].tolist()  # Số Phòng Ngủ
    num_of_floor = crawled_datas["NumOfFloor"].tolist()  # Số Tầng
    num_of_toilet = crawled_datas["NumOfToilet"].tolist()  # Số Nhà Vệ Sinh
    construction_year = crawled_datas["ConstructionYear"].tolist()  # Năm Xây Dựng
    # status_of_re = crawled_datas["StatusOfRE"].tolist()  # Tình Trạng BĐS
    # characteristics_re = crawled_datas["CharacteristicsRE"].tolist()  # Đặc Tính BĐS
    is_owner = crawled_datas["IsOwner"].tolist()  # Chính Chủ
    furniture = crawled_datas["Furniture"].tolist()  # Tình Trạng Nội Thất
    terrace = crawled_datas["Terrace"].tolist()  # Sân Thượng
    car_parking = crawled_datas["CarParking"].tolist()  # Chỗ Để Xe Hơi
    dinning_room = crawled_datas["DinningRoom"].tolist()  # Phòng Ăn
    kitchen = crawled_datas["Kitchen"].tolist()  # Nhà Bếp
    air_cond = crawled_datas["AirCond"].tolist()  # Điều Hòa
    adsl = crawled_datas["ADSL"].tolist()  # ADSL
    washing_machine = crawled_datas["WashingMachine"].tolist()  # Máy Giặt
    balcony = crawled_datas["Balcony"].tolist()  # Ban Công
    fridge = crawled_datas["Fridge"].tolist()  # Tủ Lạnh
    wifi = crawled_datas["Wifi"].tolist()  # Wifi
    pool = crawled_datas["Pool"].tolist()  # Pool
    basement = crawled_datas["Basement"].tolist()  # Tầng Hầm
    park = crawled_datas["Park"].tolist()  # Công Viên
    super_market = crawled_datas["SuperMarket"].tolist()  # Siêu Thị
    clinics = crawled_datas["Clinics"].tolist()  # Trạm Xá
    sea = crawled_datas["Sea"].tolist()  # Biển
    hospital = crawled_datas["Hospital"].tolist()  # Bệnh Viện
    church = crawled_datas["Church"].tolist()  # Nhà Thờ
    bus_station = crawled_datas["BusStation"].tolist()  # Bến Xe Buýt
    school = crawled_datas["School"].tolist()  # Trường Học
    temple = crawled_datas["Temple"].tolist()  # Chùa
    airport = crawled_datas["Airport"].tolist()  # Sân Bay
    pre_school = crawled_datas["Preschool"].tolist()  # Trường Mầm Non
    characteristics = crawled_datas["Characteristics"].tolist()  # Đặc Tính
    # price_psm = crawled_datas["PricePSM"].tolist()  # Giá / m2
    price = crawled_datas["Price"].tolist()  # Giá

    # todo: create/reconstruct CSV file to train data
    train_file = open(out_train_file_path, "w", newline='', encoding='utf-8-sig')
    test_file = open(out_test_file_path, "w", newline='', encoding='utf-8-sig')
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)
    print("n records: ", crawled_datas.shape[0])
    train_writer.writerow(["ID", "FormRE", "Province", "FloorNum", "Area", "Height", "Width", "UsableArea",
                           "FrontLength", "BackSideLength", "Direction", "BalconyDirection", "Corner", "RoadInFront",
                           "Juridical", "NumOfBed", "NumOfFloor", "NumOfToilet", "ConstructionYear", "IsOwner",
                           "Furniture", "Terrace", "CarParking", "DinningRoom", "Kitchen", "AirCond",
                           "ADSL", "WashingMachine", "Balcony", "Fridge", "Wifi", "Pool",
                           "Basement", "Park", "SuperMarket", "Clinics", "Sea", "Hospital", "Church",
                           "BusStation", "School", "Temple", "Airport", "Preschool", "Characteristics",
                           "Price"])
    test_writer.writerow(["ID", "FormRE", "Province", "FloorNum", "Area", "Height", "Width", "UsableArea",
                          "FrontLength", "BackSideLength", "Direction", "BalconyDirection", "Corner", "RoadInFront",
                          "Juridical", "NumOfBed", "NumOfFloor", "NumOfToilet", "ConstructionYear", "IsOwner",
                          "Furniture", "Terrace", "CarParking", "DinningRoom", "Kitchen", "AirCond",
                          "ADSL", "WashingMachine", "Balcony", "Fridge", "Wifi", "Pool",
                          "Basement", "Park", "SuperMarket", "Clinics", "Sea", "Hospital", "Church",
                          "BusStation", "School", "Temple", "Airport", "Preschool", "Characteristics"])

    import math
    id_rec = 0
    for rec in range(crawled_datas.shape[0]):
        # for rec in range(19000):
        # todo: drop all fields with Giá = nan
        if is_valid_record(width[rec], height[rec], area[rec], price[rec]):
            if id_rec < crawled_datas.shape[0] / 1.2:
                # if id_rec < 9000:
                train_writer.writerow(
                    [id_rec + 1, form_re[rec], province[rec], floor_num[rec], area[rec], height[rec], width[rec],
                     usable_area[rec], front_length[rec], back_side_length[rec], direction[rec],
                     balcony_direction[rec], corner[rec], road_in_front[rec], juridical[rec], num_of_bed[rec],
                     num_of_floor[rec], num_of_toilet[rec], construction_year[rec], is_owner[rec],
                     furniture[rec],
                     terrace[rec], car_parking[rec], dinning_room[rec], kitchen[rec], air_cond[rec], adsl[rec],
                     washing_machine[rec], balcony[rec], fridge[rec], wifi[rec], pool[rec], basement[rec],
                     park[rec],
                     super_market[rec], clinics[rec], sea[rec], hospital[rec], church[rec], bus_station[rec],
                     school[rec], temple[rec], airport[rec], pre_school[rec], characteristics[rec],
                     price[rec]])
            else:
                test_writer.writerow(
                    [id_rec + 1, form_re[rec], province[rec], floor_num[rec], area[rec], height[rec], width[rec],
                     usable_area[rec], front_length[rec], back_side_length[rec], direction[rec],
                     balcony_direction[rec], corner[rec], road_in_front[rec], juridical[rec], num_of_bed[rec],
                     num_of_floor[rec], num_of_toilet[rec], construction_year[rec], is_owner[rec],
                     furniture[rec],
                     terrace[rec], car_parking[rec], dinning_room[rec], kitchen[rec], air_cond[rec], adsl[rec],
                     washing_machine[rec], balcony[rec], fridge[rec], wifi[rec], pool[rec], basement[rec],
                     park[rec],
                     super_market[rec], clinics[rec], sea[rec], hospital[rec], church[rec], bus_station[rec],
                     school[rec], temple[rec], airport[rec], pre_school[rec], characteristics[rec]])

            id_rec += 1
    print("n records filtered: ", id_rec)
    train_file.close()
    test_file.close()

    # area = crawled_data["Area"].tolist()
    # print("type of area:", type(area))

    # add new columns to last of columns with empty values
    # crawled_data["MyID"] = ""

    # rename exists columns
    # crawled_data.rename(columns={'Area': 'Area (test changed)'}, inplace=True)

    # redefine data of columns

    # save new file
    # crawled_data.to_csv(input_des_path, index=False, encoding='utf-8-sig')  # encoding: 'utf-8-sig' for Vietnamese
    print("done!")
    print("Convert CSV time: ", timer() - start, "(s)")
    return


def is_valid_record(width, height, area, price):
    i = 0
    import math
    if math.isnan(width):
        i += 1
    if math.isnan(height):
        i += 1
    if math.isnan(area):
        i += 1

    if i >= 2 or math.isnan(price):
        return False
    else:
        return True


def is_belong_to_province(province_data, province_id=-1):
    # Filter for all provinces
    if province_id == -1:
        return True
    # Filter for only 1 specific province
    elif province_data == province_id:
        return True
    else:
        return False


def drop_data_and_save(data, all_data_na, saved_file_path):
    # todo: drop columns missing almost values
    # for column in all_data_na.index:
    #     if all_data_na[column] > 40:
    #         data.drop([column], axis=1, inplace=True)
    # data.to_csv(saved_file_path, index=False, encoding='utf-8-sig')
    return data


def inspect_outlier(data, inspect_features, target_features):
    import matplotlib.pyplot as plt  # Matlab-style plotting
    fig, ax = plt.subplots()
    ax.scatter(x=data[inspect_features], y=data[target_features])
    plt.title("INSPECT OUTLIERS: " + inspect_features + " - " + target_features)
    plt.ylabel(target_features, fontsize=13)
    plt.xlabel(inspect_features, fontsize=13)
    plt.show()


def analyze_target_features(data):
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    from scipy import stats

    sns.distplot(data['Price'], fit=norm)
    # print("type of: ", type(data["Giá"]))
    # print("value of: ", data["Giá"])

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['Price'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Price distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['Price'], plot=plt)
    plt.show()


def analyze_target_features_transform(data):
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    from scipy import stats

    # We use the numpy function log1p which  applies log(1+x) to all elements of the column
    data["Price"] = np.log1p(data["Price"])

    # Check the new distribution
    sns.distplot(data["Price"], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data["Price"])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Price distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data["Price"], plot=plt)
    plt.show()
