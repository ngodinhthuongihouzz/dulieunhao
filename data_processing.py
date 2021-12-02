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


def drop_id(train, test):
    # Drop the  'Id' column since it's unnecessary for  the prediction process.
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
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


def convert_num_to_string_values_test(all_data):
    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    # # Changing OverallCond into a categorical variable
    # all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    # # Year and month sold are transformed into categorical features.
    # all_data['YrSold'] = all_data['YrSold'].astype(str)
    # all_data['MoSold'] = all_data['MoSold'].astype(str)

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


# todo: ignore dummy data by labeling all categorical features to number
def label_encoding_test(all_data):
    from sklearn.preprocessing import LabelEncoder
    cols = ('LandSlope', 'LotShape', 'Street', 'Alley', 'MSSubClass',
            'MSZoning', 'LandContour', 'LotConfig', 'Utilities', 'Neighborhood')  # không chừa một cột nào còn
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


def transform_numerical_to_categorical_values(all_data):
    all_data = convert_num_to_string_values(all_data)
    all_data = label_encoding(all_data)
    return all_data


def transform_numerical_to_categorical_values_test(all_data):
    all_data = convert_num_to_string_values_test(all_data)
    all_data = label_encoding_test(all_data)
    return all_data


def load_transform_numerical_to_categorical_values(all_data):
    all_data = convert_num_to_string_values(all_data)
    all_data = label_encoding_loaded(all_data)
    return all_data


def record_load_transform_numerical_to_categorical_values(all_data):
    all_data = record_convert_num_to_string_values(all_data)
    all_data = label_encoding_loaded(all_data)
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
    print(all_data.shape)
    train = all_data[:n_train]
    test = all_data[n_train:]
    return all_data, train, test


def convert_crawled_to_input_data(crawled_path, out_file_path):
    from timeit import default_timer as timer
    start = timer()
    print("start convert_crawled_to_input_data ...")
    crawled_data = pd.read_csv(crawled_path)

    # todo: drop columns ID, Type, TypeOfRealEstate, Province, District, Ward, Street, Project, LandNum, BlockName
    crawled_data.drop(['ID'], axis=1, inplace=True)
    crawled_data.drop(['Type'], axis=1, inplace=True)
    crawled_data.drop(['TypeOfRealEstate'], axis=1, inplace=True)
    crawled_data.drop(['Province'], axis=1, inplace=True)
    crawled_data.drop(['District'], axis=1, inplace=True)
    crawled_data.drop(['Ward'], axis=1, inplace=True)
    crawled_data.drop(['Street'], axis=1, inplace=True)
    crawled_data.drop(['Project'], axis=1, inplace=True)
    crawled_data.drop(['LandNum'], axis=1, inplace=True)
    crawled_data.drop(['BlockName'], axis=1, inplace=True)

    # todo: get each row of data to list
    form_real_estate = crawled_data["FormRealEstate"].tolist()  # Loại Hình BĐS
    floor_num = crawled_data["FloorNum"].tolist()  # Tầng Số
    area = crawled_data["Area"].tolist()  # Diện Tích
    height = crawled_data["Height"].tolist()  # Chiều Dài
    width = crawled_data["Width"].tolist()  # Chiều Rộng
    usable_area = crawled_data["UsableArea"].tolist()  # DTSD
    front_length = crawled_data["FrontLength"].tolist()  # Mặt Tiền
    back_side_length = crawled_data["BackSideLength"].tolist()  # Mặt Hậu
    direction = crawled_data["Direction"].tolist()  # Hướng
    balcony_direction = crawled_data["BalconyDirection"].tolist()  # Hướng Ban Công
    corner = crawled_data["Corner"].tolist()  # Căn Góc
    road_in_front = crawled_data["RoadInFront"].tolist()  # Đường Trước Nhà
    juridical = crawled_data["Juridical"].tolist()  # Pháp Lý
    num_of_bed = crawled_data["NumOfBed"].tolist()  # Số Phòng Ngủ
    num_of_floor = crawled_data["NumOfFloor"].tolist()  # Số Tầng
    num_of_toilet = crawled_data["NumOfToilet"].tolist()  # Số Nhà Vệ Sinh
    construction_year = crawled_data["ConstructionYear"].tolist()  # Năm Xây Dựng
    status_of_re = crawled_data["StatusOfRE"].tolist()  # Tình Trạng BĐS
    characteristics_re = crawled_data["CharacteristicsRE"].tolist()  # Đặc Tính BĐS
    is_owner = crawled_data["IsOwner"].tolist()  # Chính Chủ
    furniture = crawled_data["Furniture"].tolist()  # Tình Trạng Nội Thất
    terrace = crawled_data["Terrace"].tolist()  # Sân Thượng
    car_parking = crawled_data["CarParking"].tolist()  # Chỗ Để Xe Hơi
    dinning_room = crawled_data["DinningRoom"].tolist()  # Phòng Ăn
    kitchen = crawled_data["Kitchen"].tolist()  # Nhà Bếp
    air_cond = crawled_data["AirCond"].tolist()  # Điều Hòa
    internet = crawled_data["Internet"].tolist()  # Internet (ADSL & Cáp Quang)
    washing_machine = crawled_data["WashingMachine"].tolist()  # Máy Giặt
    balcony = crawled_data["Balcony"].tolist()  # Ban Công
    fridge = crawled_data["Fridge"].tolist()  # Tủ Lạnh
    wifi = crawled_data["Wifi"].tolist()  # Wifi
    pool = crawled_data["Pool"].tolist()  # Pool
    basement = crawled_data["Basement"].tolist()  # Tầng Hầm
    super_market = crawled_data["SuperMarket"].tolist()  # Siêu Thị
    market = crawled_data["Market"].tolist()  # Chợ
    park = crawled_data["Park"].tolist()  # Công Viên
    clinics = crawled_data["Clinics"].tolist()  # Trạm Xá
    sea = crawled_data["Sea"].tolist()  # Biển
    hospital = crawled_data["Hospital"].tolist()  # Bệnh Viện
    church = crawled_data["Church"].tolist()  # Nhà Thờ
    bus_station = crawled_data["BusStation"].tolist()  # Bến Xe Buýt
    school = crawled_data["School"].tolist()  # Trường Học
    temple = crawled_data["Temple"].tolist()  # Chùa
    airport = crawled_data["Airport"].tolist()  # Sân Bay
    pre_school = crawled_data["Preschool"].tolist()  # Trường Mầm Non
    price_psm = crawled_data["PricePSM"].tolist()  # Giá / m2
    price = crawled_data["Price"].tolist()  # Giá

    # todo: create/reconstruct CSV file to train data
    file = open(out_file_path, "w", newline='', encoding='utf-8-sig')
    writer = csv.writer(file)
    print("n records: ", crawled_data.shape[0])
    writer.writerow(["Id", "Loại Hình BDS", "Tầng Số", "Diện Tích", "Chiều Dài", "Chiều Rộng", "DTSD",
                     "Mặt Tiền", "Mặt Hậu", "Hướng", "Hướng Ban Công", "Căn góc", "Đường Trước Nhà", "Pháp Lý",
                     "Số Phòng Ngủ", "Số Tầng", "Số Nhà Vệ Sinh", "Năm Xây Dựng", "Tình Trạng BĐS",
                     "Đặc Tính BĐS", "Chính Chủ", "Tình Trạng Nội Thất", "Sân Thượng", "Chỗ Để Xe Hơi",
                     "Phòng Ăn", "Nhà Bếp", "Điều Hòa", "Internet (ADSL & Cáp Quang)", "Máy Giặt", "Ban Công",
                     "Tủ Lạnh", "Wifi", "Hồ Bơi", "Tầng Hầm", "Siêu Thị", "Chợ", "Công Viên", "Trạm Xá",
                     "Biển", "Bệnh Viện", "Nhà Thờ", "Bến Xe Buýt", "Trường Học", "Chùa", "Sân Bay",
                     "Trường Mầm Non", "Giá / m2", "Giá"])
    for rec in range(crawled_data.shape[0]):
        writer.writerow([rec + 1, form_real_estate[rec], floor_num[rec], area[rec], height[rec], width[rec],
                         usable_area[rec], front_length[rec], back_side_length[rec], direction[rec],
                         balcony_direction[rec], corner[rec], road_in_front[rec], juridical[rec], num_of_bed[rec],
                         num_of_floor[rec], num_of_toilet[rec], construction_year[rec], status_of_re[rec],
                         characteristics_re[rec], is_owner[rec], furniture[rec], terrace[rec], car_parking[rec],
                         dinning_room[rec], kitchen[rec], air_cond[rec], internet[rec], washing_machine[rec],
                         balcony[rec], fridge[rec], wifi[rec], pool[rec], basement[rec], super_market[rec],
                         market[rec], park[rec], clinics[rec], sea[rec], hospital[rec], church[rec], bus_station[rec],
                         school[rec], temple[rec], airport[rec], pre_school[rec], price_psm[rec], price[rec]])
    file.close()

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
    print("Convert CSV(Edited) time: ", timer() - start, "(s)")
    return


def inspect_distribution(data, inspect_features, target_features):
    import matplotlib.pyplot as plt  # Matlab-style plotting
    fig, ax = plt.subplots()
    ax.scatter(x=data[inspect_features], y=data[target_features])
    plt.title("DISTRIBUTION: " + inspect_features + " - " + target_features)
    plt.ylabel(target_features, fontsize=13)
    plt.xlabel(inspect_features, fontsize=13)
    # plt.show()


def analyze_target_features(data):
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    from scipy import stats

    sns.distplot(data['Giá'], fit=norm)  # todo: drop all fields with Giá = nan

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['Giá'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Price distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data['Giá'], plot=plt)
    plt.show()
