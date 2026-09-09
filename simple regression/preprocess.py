import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the original dataset (assumes census_income_dataset.csv or adult.csv with headers)
data = pd.read_csv("adult.csv", names=[
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gains", "capital-loss", "hours-per-week", "native-country", "target"
], skipinitialspace=True, na_values="?")

# 2. Drop the 5 irrelevant columns
data = data.drop(["fnlwgt", "education", "relationship", "sex", "race"], axis=1)

# 3. Use LabelEncoder for categorical variables and target
enc = LabelEncoder()
features_to_convert = ["workclass", "marital-status", "occupation", "native-country", "target"]

for i in features_to_convert:
    data[i] = enc.fit_transform(data[i].astype('str'))

# 4. Save preprocessed dataset matching the expected repository structure
data.to_csv("census_income_dataset_preprocessed.csv", index=False)
print("Saved clean dataset shape:", data.shape)