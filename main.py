import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # For numerical attributes
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("standardization", StandardScaler())
    ])

    # For categorical attributes
    cat_pipeline = Pipeline([
        ("encoder_1hot", OneHotEncoder())
    ])

    # Full pipeline for both numerical and categorical attributes
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Train the model:

    # Read the dataset
    data = pd.read_csv('housing.csv')

    # Split the dataset into train and test data
    data['income_cat'] = pd.cut(data['median_income'], 
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index] # keep aside will work on this later

    for sett in (strat_train_set, strat_test_set):
        sett.drop('income_cat', axis=1, inplace=True)

    strat_test_set.to_csv('input.csv', index=False)

    # We will work on the copy of strat_train_set
    df = strat_train_set.copy()

    # Separate features and labels from train set
    housing_features = df.drop('median_house_value', axis=1)
    housing_label = df['median_house_value'].copy()

    # Separate and list numerical and categorical columns
    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline =  build_pipeline(num_attribs, cat_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_label)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Congrats! Model is successfully trained.")

else:
    # We will do the inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv('predicted_output.csv', index=False)
    print("Congrats! predictions are made successfully and is saved to predicted_output.csv")
