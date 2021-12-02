import data_processing as dp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# convert data from raw data (crawled data) and save it to new csv input file
dp.convert_crawled_to_input_data("input/crawled/Demo.csv", "input/crawled/Demo[Edited].csv")
# read new csv input file
data = pd.read_csv("input/crawled/Demo[Edited].csv")
# todo: delete_outliers, check later for each features and remove unbalanced data if any
dp.inspect_distribution(data, 'Diện Tích', 'Giá')
dp.inspect_distribution(data, 'DTSD', 'Giá')
# inspect target variable (Giá)
dp.analyze_target_features(data)
# check missing values (nan/null values) of columns data and show first 100 features (if any)
dp.check_remaining_missing_values(data)
# imputing missing values
data["Nhà Thờ"] = data["Nhà Thờ"].fillna("None")  # nan values means no "Nhà Thờ" over there
# data[""] = data[""]

print("...")

