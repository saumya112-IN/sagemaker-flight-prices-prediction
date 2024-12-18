#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries

# In[1]:


pip install --upgrade pandas jupyter ipython


# In[2]:


get_ipython().system(' pip install xgboost')


# In[3]:


get_ipython().system(' pip install feature-engine')


# In[5]:


import os

import boto3

import pickle

import warnings

import numpy as np

import pandas as pd

import xgboost as xgb

import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	MinMaxScaler,
	PowerTransformer,
	FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    HyperparameterTuner
)


# ## 2. Display Settings

# In[6]:


pd.set_option("display.max_columns", None)


# In[7]:


sklearn.set_config(transform_output="pandas")


# In[8]:


warnings.filterwarnings("ignore")


# ## 3. Read Datasets

# In[9]:


train = pd.read_csv("train.csv")
train


# In[10]:


val = pd.read_csv("val.csv")
val


# In[11]:


test = pd.read_csv("test.csv")
test


# ## 4. Preprocessing Operations

# In[12]:


# airline
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

#doj
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

# source & destination
location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(X):
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# dep_time & arrival_time
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:, col]).dt.hour
        for col in columns
    })

    return (
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:, col].between(morning, noon, inclusive="left"),
                 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                 X_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon", "evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# duration
class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma
        
    def fit(self, X, y=None):
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (
                X
                .loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }

        return self


    def transform(self, X):
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)
    

def duration_category(X, short=180, med=400):
    return (
        X
        .assign(duration_cat=np.select([X.duration.lt(short),
                                    X.duration.between(short, med, inclusive="left")],
                                    ["short", "medium"],
                                    default="long"))
        .drop(columns="duration")
    )

def is_over(X, value=1000):
    return (
        X
        .assign(**{
            f"duration_over_{value}": X.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),
    ("scaler", PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),
    ("part2", duration_pipe2),
    ("part3", FunctionTransformer(func=is_over)),
    ("part4", StandardScaler())
])

duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", duration_union)
])

# total_stops
def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("", FunctionTransformer(func=is_direct))
])

# additional_info
info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
("part1", info_pipe1),
("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
("union", info_union)
])

# column transformer
column_transformer = ColumnTransformer(transformers=[
("air", air_transformer, ["airline"]),
("doj", doj_transformer, ["date_of_journey"]),
("location", location_transformer, ["source", 'destination']),
("time", time_transformer, ["dep_time", "arrival_time"]),
("dur", duration_transformer, ["duration"]),
("stops", total_stops_transformer, ["total_stops"]),
("info", info_transformer, ["additional_info"])
], remainder="passthrough")

# feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
estimator=estimator,
scoring="r2",
threshold=0.1
) 

# preprocessor
preprocessor = Pipeline(steps=[
("ct", column_transformer),
("selector", selector)
])


# In[13]:


preprocessor.fit(
    train.drop(columns="price"),
    train.price.copy()
)


# In[14]:


preprocessor.transform(train.drop(columns="price"))


# ## 4. Preprocess Data and Upload to Bucket

# In[15]:


BUCKET_NAME = "sagemaker-flights-prices-bucket"

DATA_PREFIX = "data"


# In[16]:


def get_file_name(name):
    return f"{name}-pre.csv"


# In[17]:


def export_data(data, name, pre):
    # split data into X and y subsets
    X = data.drop(columns="price")
    y = data.price.copy()
    
    # transformation
    X_pre = pre.transform(X)
    
    # exporting
    file_name = get_file_name(name)
    (
        y
        .to_frame()
        .join(X_pre)
        .to_csv(file_name, index=False, header=False)
    )


# In[18]:


def upload_to_bucket(name):
    file_name = get_file_name(name)
    
    (
        boto3
        .Session()
        .resource("s3")
        .Bucket(BUCKET_NAME)
        .Object(os.path.join(DATA_PREFIX, f"{name}/{name}.csv"))
        .upload_file(file_name)
    )


# In[19]:


def export_and_upload_bucket(data, name, pre):
    export_data(data, name, pre)
    upload_to_bucket(name)


# In[20]:


export_and_upload_bucket(train, "train", preprocessor)


# In[21]:


export_and_upload_bucket(val, "val", preprocessor)


# In[22]:


export_and_upload_bucket(test, "test", preprocessor)


# ## 5. Model and Hyperparameter Tuning Set-up

# In[23]:


session = sagemaker.Session()
region_name = session.boto_region_name


# In[24]:


output_path = f"s3://{BUCKET_NAME}/model/output"


# In[25]:


model = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region_name, "1.2-1"),
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m4.xlarge",
    volume_size=5,
    output_path=output_path,
    use_spot_instances=True,
    max_run=300,
    max_wait=600,
    sagemaker_session=session
)


# In[26]:


model.set_hyperparameters(
    objective="reg:linear",
    num_round=10,
    eta=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.1
)


# In[27]:


hyperparameter_ranges = {
    "eta": ContinuousParameter(0.05, 0.2),
    "alpha": ContinuousParameter(0, 1),
    "max_depth": IntegerParameter(3, 5)
}


# In[28]:


tuner = HyperparameterTuner(
    estimator=model,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges=hyperparameter_ranges,
    strategy="Bayesian",
    objective_type="Minimize"
)


# ## 6. Data Channels

# In[29]:


def get_data_channel(name):
    bucket_path = f"s3://{BUCKET_NAME}/{DATA_PREFIX}/{name}"
    return TrainingInput(bucket_path, content_type="csv")


# In[30]:


train_data_channel = get_data_channel("train")
train_data_channel


# In[31]:


val_data_channel = get_data_channel("val")


# In[32]:


data_channels = {
    "train": train_data_channel,
    "validation": val_data_channel
}


# ## 7. Train and Tune the Model

# In[33]:


tuner.fit(data_channels)


# ## 8. Model Evaluation

# In[35]:


with open("xgboost-model", "rb") as f:
    best_model = pickle.load(f)
    
best_model


# In[36]:


def evaluate_model(name):
    file_name = get_file_name(name)
    data = pd.read_csv(file_name)
    
    X = xgb.DMatrix(data.iloc[:, 1:])
    y = data.iloc[:, 0].copy()
    
    pred = best_model.predict(X)
    
    return r2_score(y, pred)


# In[37]:


evaluate_model("train")


# In[38]:


evaluate_model("val")


# In[39]:


evaluate_model("test")


# In[ ]:




