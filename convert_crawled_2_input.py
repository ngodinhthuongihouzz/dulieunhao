import data_processing as dp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# convert data from raw data (crawled data) and save it to new csv input file
dp.convert_crawled_to_input_data(dp.get_all_csv_in_directory('input/train/train2021122*.csv'),
                                 "input/train/train20211207_110259[train].csv",
                                 "input/test/train20211207_110259[test].csv")

# read new csv input file
# data = pd.read_csv("input/crawled/train20211207_110259[Edited].csv")

# check missing values (nan/null values) of columns data and show first 100 features (if any)
# data = dp.check_remaining_missing_values_and_remove(data)
# fil unset data's columns
# data = dp.fill_unset_rows(data)
# data = dp.check_remaining_missing_values_and_remove(data)

# imputing missing values
# data["Church"] = data["Church"].fillna("None")  # nan values means no "Nhà Thờ" over there
# data[""] = data[""]

# todo: delete_outliers, check later for each features and remove unbalanced data if any for number only
# dp.inspect_outlier(data, 'Diện Tích', 'Giá')
# dp.inspect_outlier(data, 'DTSD', 'Giá')
# dp.inspect_outlier(data, 'UsableArea', 'Price')
# dp.inspect_outlier(data, 'Trường Học', 'Giá') # for boolean didn't work

# inspect target variable (Giá)
# dp.analyze_target_features(data)
# dp.analyze_target_features_transform(data)

print("...")
