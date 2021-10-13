import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm, skew  # for some statistics


def read_input(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


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
    print(missing_data.head())


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


def label_encoding(all_data):
    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # inspect shape
    print('Shape all_data: {}'.format(all_data.shape))

    return all_data


def transform_numerical_to_categorical_values(all_data):
    all_data = convert_num_to_string_values(all_data)
    all_data = label_encoding(all_data)
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
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness.head(10))
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
    return all_data


def getting_new_train_test(all_data, n_train):
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)
    train = all_data[:n_train]
    test = all_data[n_train:]
    return all_data, train, test

