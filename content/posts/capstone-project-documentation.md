---
weight: 10
title: "Capstone Project Documentation"
date: 2023-06-26T21:29:01+08:00
description: "Documentation of the project and results"
tags: ["bike", "model", "prediction","machine learning"]
type: post
showTableOfContents: true
---

## Introduction

The Bicing service, since its start in 2007, has become a great way to move around the city, leveraging the great bike lane network of the city. Thhroughout the years

The new Bicing service includes more territorial coverage, an increase in the number of bicycles, mixed stations for conventional and electric bicycles, new and improved types of stations and bicycles (safety, anchorage, comfort), extended schedules and much more!

## Goals of the project

There are two main objective in this project:

- Predict the number of free docks given the historical data (Docks Availability Percent).

- Explore new places where stations are needed.

- Explore how different events affect availability.


## Get the Data

According with the project we have the next sources:

- The bicing stations status
- The bicing stations information
- The weather of the city of Barcelona

The Bicing stations status and information of the city of Barcelona were downloaded from [Open Data BCN](https://opendata-ajuntament.barcelona.cat) and the weather information to join with bicing stations were downloaded from [Meteo Cat](https://www.meteo.cat).

### a. Download the Data

The following script was used to download the data of [Open Data BCN](https://opendata-ajuntament.barcelona.cat) by year and month:

```bash
import os

i2m = list(zip(range(1,13), ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']))
for year in [2022, 2021, 2020, 2019]:
    for month, month_name in i2m:        
        os.system(f"wget 'https://opendata-ajuntament.barcelona.cat/resources/bcn/BicingBCN/{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")
        os.system(f"7z x '{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")
        os.system(f"rm '{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")

```

The following script was used to download the data of [Meteo Cat](https://www.meteo.cat) to join and use in the patterns of the availability to the stations:

```python
# CODE TO RETRIEVE DATA FROM THE WEATHER API AND STORE IT IN A CSV FILE
# FROM 2019/01/01 TO 2023/03/31

import requests
import time
import datetime as dt
import csv 

key = 'enJH8FUX2z5Ar7NSCJvYI8pAIuDW0XDV9nbSkEMj'

start_date = dt.date(2019, 1, 1)
end_date = dt.date(2023, 3, 31)

assert start_date < end_date, 'Start date must be before end date'

delta = dt.timedelta(days=1)

freq = 1/20

day = start_date

columns = ['timestamp', 'mm_precip', 'temperature']

with open('weather_data/weather.csv', mode = 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()

    while day <= end_date:

        day_string = day.strftime('%Y/%m/%d')

        url = 'https://api.meteo.cat/xema/v1' + '/estacions/mesurades/X8/' + day_string 

        print(day_string)

        headers = {'Accept': 'application/json', 'X-API-KEY': key}

        data = requests.get(url, headers=headers).json()[0]

        # the codi variables, 35 and 32, correspond to precipitation and temperature respectively. The json file retrieved
        # contains information on many more variables, but we are only interested in these two.

        precipitation = [data['variables'][i]['lectures'] for i in range(len(data['variables'])) if data['variables'][i]['codi'] == 35][0]
        temperature = [data['variables'][i]['lectures'] for i in range(len(data['variables'])) if data['variables'][i]['codi'] == 32][0]

        date_variables = [{'timestamp':int(dt.datetime.strptime(d['data'], '%Y-%m-%dT%H:%MZ').timestamp()), 'mm_precip':d['valor']} for d in precipitation]

        for i in range(len(date_variables)):
            date_variables[i]['temperature'] = temperature[i]['valor']

        writer.writerows(date_variables)

        day += delta

        time.sleep(freq)
```

### b. Consolidate the Data

The consolidation of the Data was built in Tableau using Tableau Prep which give us more flexibility to join, filter and build the next structure according with our metadata-sample-submission.csv

#### Metadata Sample Submission

![Metadata Sample Submission](/capstone-project/metadata-sample-submission.png)

## Discover and visualize the data to gain insights

Data consolidation, visualization and analysis with Tableau

### a. Tableau Workflow

The first flow in Tableau Prep let you consolidate the bicing station status files of all years and create new columns like year, month, day and hour. 
The flow has two sub-flows because is necessary applied the same process to bicing station 2023 files. Both flows create new files with a hyper format which is easier to manage and control the data.

![First Tableau flow](/capstone-project/first-tableau-flow.png)

The second flow in Tableau Prep get the hyper files of bicing station status (2019-2022) to join with the last bicing station information (March, 2023) to get the extra information like longitude, latitude, name, capacity and postcode.

![Second Tableau flow](/capstone-project/second-tableau-flow.png)

The same process is applied to the bicing station status 2023 but filtering docks_availability and bike_availability. These fields will be evaluated and predicted by the models.

![Third Tableau flow](/capstone-project/third-tableau-flow.png)

Finally, the last flow transform the input data of the preview flow using a aggregation to the level year, month, day and hour calculating average of the rest of fields and creating four additional fields with the percent of docks availability in the four hours before.

![Fourth Tableau flow](/capstone-project/fourth-tableau-flow.png)

Before training and testing, the data is analyzed to identify patterns, outliers and to visualize relevant features.

### b. Analysis in Tableau

Visual analysis of data constructed based on the final output to identify outliers, COVID behaviors and relevant characteristics before training and testing with machine learning models

![Bicing Data Analysis](/capstone-project/bicing-data-analysis.png)

With all the consolidate information we start to clean and processes the data to prepare our dataframes of training and testings.

## Prepare the data for Machine Learning algorithms

### a. Importation of data in Colab

Importation of Bicing Data and Weather of our Tableau workflow and meteo script in python

```python

import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from dask_ml.preprocessing import StandardScaler, MinMaxScaler
from dask_ml.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from IPython.display import Markdown
import json

####################################################################################################
# To run in colab

station_dataframe = dd.read_csv('/content/drive/MyDrive/CapstoneProject/data_bicing_joined_HX_23.csv', assume_missing=True, delimiter=';')

weather_dataframe = dd.read_csv('/content/drive/MyDrive/CapstoneProject/weather.csv', assume_missing=True, delimiter=',')

data_2_predict = dd.read_csv('/content/drive/MyDrive/CapstoneProject/metadata_sample_submission.csv', delimiter=',')

####################################################################################################

```
### b. Data Cleaning and Filtering

- Initial cleaning to go from the format that the Tableau merger outputs to the one desired for the model
- The objective of this first step is to leave the training data in the same format as the submission one
- Drop all rows with out of service stations and year 2020, to avoid COVID effects on our training data

```python

station_dataframe = station_dataframe.loc[station_dataframe['status'] == 'IN_SERVICE']

##################################################################################################

train_df = station_dataframe.loc[(station_dataframe['year'] != 2020) & (station_dataframe['year'] != 2023)]

train_df = train_df[['station_id', 'lat', 'lon', 'year', 'month', 'day', 'hour', '% Docks Availlable',  '% Docks Available H-4','% Docks Available H-3', '% Docks Available H-2', '% Docks Available H-1']]
train_df = train_df.rename(columns={'% Docks Availlable': 'percentage'})
for i in range(1, 5):
    train_df = train_df.rename(columns={f'% Docks Available H-{i}': f'ctx-{i}'})

# Print the head of the updated DataFrame
train_df = train_df[['station_id', 'lat', 'lon', 'year', 'month', 'day', 'hour', 'ctx-4', 'ctx-3', 'ctx-2',	'ctx-1', 'percentage']]

train_df = train_df.reset_index()

##################################################################################################

validation_df = station_dataframe.loc[(station_dataframe['year'] == 2023)]

validation_df = validation_df[['station_id', 'lat', 'lon', 'year', 'month', 'day', 'hour', '% Docks Availlable',  '% Docks Available H-4','% Docks Available H-3', '% Docks Available H-2', '% Docks Available H-1']]
validation_df = validation_df.rename(columns={'% Docks Availlable': 'percentage'})
for i in range(1, 5):
    validation_df = validation_df.rename(columns={f'% Docks Available H-{i}': f'ctx-{i}'})

# Print the head of the updated DataFrame
validation_df = validation_df[['station_id', 'lat', 'lon', 'year', 'month', 'day', 'hour', 'ctx-4', 'ctx-3', 'ctx-2',	'ctx-1', 'percentage']]

validation_df = validation_df.reset_index()


```

### c. Processing pipelines

Apart from the preprocessing done using tableau to create the initial dataframe, everything else will be done using sklearn pipelines. 

Creation of the 3 pipelines:
- train_preparator: to prepare the training data, not used for the submission data as we already have location data
- submisison_preparator: to prepare the submission data, not used for the training data as we don't have location data --> Not a problem since we are not fitting to any data, just transforming with data we already have
- scaling_pipeline: to fill and scale certain column in the data, used for both training and submission data --> In this case, we are fitting to the training data, and transforming both training and submission data


```python

train_preparator = Pipeline([
    ('weather_merge', weather_merge_transformer(weather_df=weather_prepped)),  # weather_merge transformer already includes the creation of the datetime column
    ('extra_time_info', FunctionTransformer(func=extra_time_info)),
    ('hour_selector', hour_selector_transformer(hour_list=[4, 9, 14, 19, 23])), # This obviously restricts the training set, but it prevents
                                                                                # the model from seeing data labels before predicting them afterwards
                                                                                # It also prevents running out of RAM. A better approach would need to sample all
                                                                                # all hours, but it is quite more complicated than this

    ('time_normalization', time_norm_transformer(columns=['month', 'day', 'hour'])),
    ('station_id_dropper', station_id_dropper())
])

val_preparator = Pipeline([
    ('weather_merge', weather_merge_transformer(weather_df=weather_prepped)),  # weather_merge transformer already includes the creation of the datetime column
    ('extra_time_info', FunctionTransformer(func=extra_time_info)),
    ('hour_selector', hour_selector_transformer(hour_list=[i for i in range(24)])), # For the validation part, we will leave one dataset with all the hours and another one
                                                                                    # with just the four that appear in the training set to evaluate how this decision affects results
    ('time_normalization', time_norm_transformer(columns=['month', 'day', 'hour'])),
    ('station_id_dropper', station_id_dropper())
])

submission_preparator = Pipeline([
    ('weather_merge', weather_merge_transformer(weather_df=weather_prepped)),  # weather_merge transformer already includes the creation of the datetime column'
    ('extra_time_info', FunctionTransformer(func=extra_time_info)),
    ('hour_selector', hour_selector_transformer(hour_list=[i for i in range(24)])),
    ('time_normalization', time_norm_transformer(columns=['month', 'day', 'hour'])),
    ('station_loc', station_loc_transformer(id_lat_lon=final_loc)),
    ('station_id_dropper', station_id_dropper())
])



total_columns = ['index','lat', 'lon', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1',
       'temperature', 'mm_precip', 'temperature-1', 'mm_precip-1',
       'temperature-2', 'mm_precip-2', 'temperature-3', 'mm_precip-3',
       'temperature-4', 'mm_precip-4', 'is_weekend', 'timeframe1',
       'timeframe2', 'timeframe3', 'timeframe4', 'timeframe5', 'cos_month',
       'sin_month', 'cos_day', 'sin_day', 'cos_hour', 'sin_hour',
       'year_normed']


to_fill = ['index', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1', 'is_weekend', 'timeframe1',
                   'timeframe2', 'timeframe3', 'timeframe4', 'timeframe5', 'cos_month',
                   'sin_month', 'cos_day', 'sin_day', 'cos_hour', 'sin_hour',
                   'year_normed']


to_fill_and_scale = ['lat', 'lon', 'temperature', 'mm_precip', 'temperature-1', 'mm_precip-1',
       'temperature-2', 'mm_precip-2', 'temperature-3', 'mm_precip-3',
       'temperature-4', 'mm_precip-4']


filler = Pipeline([
    ('nan_filler', SimpleImputer(missing_values = np.nan, strategy='median'))
])

filler_scaler= Pipeline([
    ('nan_filler', SimpleImputer(missing_values = np.nan, strategy='median')),
    ('min_max', MinMaxScaler(feature_range = (-1,1)))
])

scaling_pipeline = ColumnTransformer([
    ('fill', filler, to_fill),
    ('fill_scale', filler_scaler, to_fill_and_scale)
])

```

### d. Classes and Functions

Each pipeline is support by a set of functions and classes to structure the data and features

```python

def weather_prep(weather_df:dd) -> dd:

    # weather dataframe preparator to leave it in the desired format to merge with the station dataframe
    # We convert the half hourly data to hourly data by averaging temperature and summing precipitation
    # We also add a column with the datetime in order to merge the dataframes

    weather = weather_df.copy()

    weather = weather.groupby(weather.index//2).mean()

    weather['mm_precip'] = weather['mm_precip']*2
    weather['timestamp'] = (weather['timestamp']-900).astype(int)
    weather['datetime'] = weather['timestamp'].map(lambda x: pd.to_datetime(x, unit='s'))


    weather['timestamp'] = weather['timestamp'].astype(int)

    weather['datetime'] = weather['timestamp'].map(lambda x: pd.to_datetime(x, unit='s'))

    weather = weather.drop_duplicates(subset=['timestamp'])

    return weather

def weather_merge(weather_df: dd, station_data: dd) -> dd:
    weather = weather_df.copy().compute()
    stations = station_data.copy().compute()

    if 'year' not in stations.columns:  # the submission df does not contain year, and we need will use that to predict
        stations['year'] = 2023

    stations[['year', 'month', 'day', 'hour']] = stations[['year', 'month', 'day', 'hour']].astype(int)

    stations['datetime'] = pd.to_datetime(stations['year'].astype(str) + '-' +
                                        stations['month'].astype(str) + '-' +
                                        stations['day'].astype(str) + ' ' +
                                        stations['hour'].astype(str) + ':00:00')

    weather['datetime'] = pd.to_datetime(weather['datetime'])

    # Sorting dataframes by datetime for the asof merge
    weather = weather.sort_values('datetime')
    stations = stations.sort_values('datetime')

    # Merge for the current datetime
    stations = pd.merge_asof(stations, weather[['datetime', 'temperature', 'mm_precip']], left_on='datetime', right_on='datetime', direction='backward')

    # Iterate to merge for the preceding datetimes
    for i in range(1, 5):
        df_weather_shifted = weather.copy()
        df_weather_shifted['datetime'] += pd.Timedelta(hours=i) # shift weather data i hours forward
        stations = pd.merge_asof(stations, df_weather_shifted[['datetime', 'temperature', 'mm_precip']], left_on='datetime', right_on='datetime', direction='backward', suffixes=('', f'-{i}'))

    return dd.from_pandas(stations, npartitions=5)  # Convert back to Dask DataFrame


####################################################################################################
####################################################################################################

class weather_merge_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, weather_df:dd, merging_function:callable=weather_merge):
        self.weather_df = weather_df
        self.func = merging_function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(self.weather_df, X)

####################################################################################################


def extra_time_info(df:dd) -> dd:

    def is_weekend(day_of_week):
        return 1 if day_of_week >= 5 else 0

    # Create a column to distinguish id ii's a weekend or not
    df['is_weekend'] = df['datetime'].dt.dayofweek.map(is_weekend, meta=('is_weekend', 'int64'))

    # Since we will only be working with 5 hours, we add 5 binary columns to indicate in which part of the day we are
    # to sort of solve the problem of the missing hours for training

    df['timeframe1'] = df['datetime'].dt.hour.map(lambda x: 1 if x <= 4 else 0, meta=('timeframe1', 'int64'))
    df['timeframe2'] = df['datetime'].dt.hour.map(lambda x: 1 if x >= 5 and x <=9 else 0, meta=('timeframe1', 'int64'))
    df['timeframe3'] = df['datetime'].dt.hour.map(lambda x: 1 if x >= 10 and x <=14 else 0, meta=('timeframe1', 'int64'))
    df['timeframe4'] = df['datetime'].dt.hour.map(lambda x: 1 if x >= 15 and x <=19 else 0, meta=('timeframe1', 'int64'))
    df['timeframe5'] = df['datetime'].dt.hour.map(lambda x: 1 if x >= 20 else 0, meta=('timeframe1', 'int64'))

    df = df.drop(['datetime'], axis = 1)

    return df

####################################################################################################


def station_loc(id_lat_lon:dd, df:dd) -> dd: #this should be applied for the 2023 march dataset to predict from station_id-location paris from february 2023

    # station location adder from the february 2023 dataset to substitute station_id for locations

    assert all(item in list(id_lat_lon.columns) for item in ['station_id', 'lat', 'lon']), 'id_lat_lon must contain station_id, lat and lon columns'
    id_locator = id_lat_lon.copy()
    data = df.copy()
    id_locator = id_locator.drop_duplicates(subset=['station_id'])

    data = data.merge(id_locator[['station_id', 'lat', 'lon']], on='station_id', how='left')

    return data

class station_loc_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_lat_lon:dd, merging_function:callable=station_loc):
        self.id_lat_lon = id_lat_lon
        self.func = merging_function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(self.id_lat_lon, X)

####################################################################################################



def hour_selector(df:dd, hour_list:list) -> dd:

    # Hour selector function to select only the hours we want to train on
    # in our case we will only train on 5 hours of the day, but a more complicated function
    # that would cover all hours could also be constructed

    data = df.copy()
    data = data.loc[data['hour'].isin(hour_list)]
    return data

class hour_selector_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, hour_list:list):
        self.hour_list = hour_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return hour_selector(X, self.hour_list)

####################################################################################################


def time_norm(df:dd, columns:list) -> dd:
    # Time normalizer function to normalize the time columns to a 0-1 scale with periodicity
    # This is done by applying a sin and cos transformation to the columns

    data = df.copy()
    for col in columns:
        data['cos_'+col] = np.cos(2*np.pi*data[col]/data[col].max())
        data['sin_'+col] = np.sin(2*np.pi*data[col]/data[col].max())
        data = data.drop([col], axis=1)

    data['year_normed'] = (data['year']-2019)/4
    data = data.drop(['year'], axis = 1)

    return data

class time_norm_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list, norm_function: callable=time_norm):
        self.columns = columns
        self.func = norm_function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X, self.columns)

####################################################################################################

class station_id_dropper(BaseEstimator, TransformerMixin):

    # Station id dropper function to drop the station_id column from the dataset at the end of the pipeline
    # prediction will be done on the location of the stations, not on the id

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(['station_id'], axis=1)
		
```

### e. Prepare all data and load data that has been saved

The following code load the data has been saved in the prepped_data folder, so there is no need to run this code again. We will just load it from our folder to work with the models

We have 3 datasets. The first is to train, the second consider only a range of hours to not overlap the data with the columns and the last dataset is our test which is loaded in Kaggle

```python

X_submission = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_submission.csv')
X_train = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_train.csv', delimiter = ',').drop('index', axis = 1)
y_train = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_train.csv', delimiter = ',')

# Validation taked 5 hours
X_val = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_val.csv', delimiter = ',').drop('index', axis = 1)
y_val = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_val.csv', delimiter = ',')

# Validation taked 24 hours
X_val24 = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_val24.csv', delimiter = ',').drop('index', axis = 1)
y_val24 = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_val24.csv', delimiter = ',')

```

#### Training Dataset

![X-Train dataset](/capstone-project/x-train.png)

#### Validation Dataset (5 hr)

![X-Val dataset](/capstone-project/x-val.png)

#### Validation Dataset with all hours (24 hr)

![X-Val24 dataset](/capstone-project/x-val24.png)

#### Submission Dataset (Testing Dataset)

![X-Submission dataset](/capstone-project/x-submission.png)

## Model Selection and Training

For this project we will use 2 different models. The first one is an xgboost regressor, that has been chosen following other similar projects to this one and that has outperformed with less training times other similar ensemble models. Then, a neural network containing LSTM layers is used to capture the temporal features and their ordering.

### a. XGBregressor

We will first use a data subset to perform a grid search on the parameter space to find the ones that fit the best our xgboost model.
 
#### RandomizedSearchCV and GridSearchCV

We apply both techniques to find the best parameters to our model after to transform our data with the pipelines

```python

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import json
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

gs_subset = X_train[X_train['year_normed'] == 0.75].sample(frac = 0.005, random_state = 5) # to restrict ourselves to more relevant data, we focus on 2022

y_subset = y_train.loc[gs_subset.index]


space = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0],
    'reg_alpha': [0, 0.1, 1.0],
    'n_estimators': [30, 50, 100, 200]
}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state = 40)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=space, scoring='neg_mean_squared_error', cv=2, verbose=2)

grid_search.fit(gs_subset, y_subset)

```

Best parameters result

![Best Parameters GridSearch](/capstone-project/grdsearch-best-parameters.png)

Now that we have the best parameters for our subset of data we train the model. We also use the validation dataset to monitor the performance of the model during training to look for over-fitting curves

```python

xgb_model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, gamma=0.1, n_jobs=-1, random_state=123, eval_metric = 'rmse', early_stopping_rounds = 30)

with open('/content/drive/MyDrive/CapstoneProject/xgboost models/best_params.json', 'r') as f:
  best_params = json.load(f)

xgb_model.set_params(**best_params)

evalset = [(X_train, y_train), (X_val, y_val), (X_val24, y_val24)]

xgb_model.fit(X_train, y_train, eval_set = evalset)

y_pred = xgb_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_true = y_val, y_pred = y_pred))

print('rmse:', rmse)

```

Once we have trained the model, we are interested in knowing how the loss evolved during training, as well as what variables were the most relevant to predict the percentage of available docks

#### Loss Function Graphic

![Loss Function graphic](/capstone-project/loss-function-xgboost.png)

#### Most Relevant Features

![Feature Importances Graphic](/capstone-project/feature-importance-xgboost.png)

The feature important plot is really interesting. From it we can conclude that:

- A expected, the previous hours percentage gives a lot of info about how many docks will be available in the following hours.
- The location of the station are the next most relevant features. The usage of the service is not homogeneous across the city and this impacts to availability of docks
- timeframe2, corresponding to peak hours in the morning has quite a bit of weight. The stations network probably has the most activity in these hours
- the hour also affects quite a bit the final %, as expected --> Since we are only training with
- The weather does not even appear in the graph. We expected way more weight from those variables, especially rain. We guess that since Barcelona is not a very rainy city, the model does not see many registers with rain and so it end up ignoring completely the variable

### b. LSTM

Apart from our xgbregressor model, and since our data contains historical data, we have also tried to study the prediction problem as a time-series analysis by working with long short-term memory layers on a neural network to try and exploit the temporal ordering of the information. Since not all the variables are historical, that have built a time series with just the one that are historical context, and we have treated the rest with a fully connected network. By using this approach our intention is to capture not only the static context of the data, but also the past that leads to each station's state.

```python

import numpy as np
import pandas as pd
import tensorflow as tf
# Assuming your dataframe is named 'df'
# Extract the static data, the target and the time series data in seperate arrays

X_submission = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_submission.csv')
X_train = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_train.csv', delimiter = ',').drop('index', axis = 1)
y_train = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_train.csv', delimiter = ',')

X_val = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_val.csv', delimiter = ',').drop('index', axis = 1)
y_val = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_val.csv', delimiter = ',')

X_val24 = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/X_val24.csv', delimiter = ',').drop('index', axis = 1)
y_val24 = pd.read_csv('/content/drive/MyDrive/CapstoneProject/prepped_data/y_val24.csv', delimiter = ',')

#################################################################################


static_columns = ['is_weekend', 'timeframe1',
       'timeframe2', 'timeframe3', 'timeframe4', 'timeframe5', 'cos_month',
       'sin_month', 'cos_day', 'sin_day', 'cos_hour', 'sin_hour',
       'year_normed', 'lat', 'lon', 'temperature', 'mm_precip']


X_train_static = X_train[static_columns].values

X_train_ts = np.stack((X_train[['temperature-1', 'mm_precip-1', 'ctx-1']].values,
                       X_train[['temperature-2', 'mm_precip-2', 'ctx-2']].values,
                       X_train[['temperature-3', 'mm_precip-3', 'ctx-3']].values,
                       X_train[['temperature-4', 'mm_precip-4', 'ctx-4']].values),
                       axis = 1)
train_target = y_train[['percentage']].values

#################################################################################

X_val_static = X_val[static_columns].values

X_val_ts = np.stack((X_val[['temperature-1', 'mm_precip-1', 'ctx-1']].values,
                     X_val[['temperature-2', 'mm_precip-2', 'ctx-2']].values,
                     X_val[['temperature-3', 'mm_precip-3', 'ctx-3']].values,
                     X_val[['temperature-4', 'mm_precip-4', 'ctx-4']].values),
                     axis = 1)
val_target = y_val[['percentage']].values

#################################################################################

X_val24_static = X_val24[static_columns].values

X_val24_ts = np.stack((X_val24[['temperature-1', 'mm_precip-1', 'ctx-1']].values,
                       X_val24[['temperature-2', 'mm_precip-2', 'ctx-2']].values,
                       X_val24[['temperature-3', 'mm_precip-3', 'ctx-3']].values,
                       X_val24[['temperature-4', 'mm_precip-4', 'ctx-4']].values),
                     axis = 1)
val24_target = y_val24[['percentage']].values

################################################################################


X_sub_static = X_submission[static_columns].values

X_sub_ts = np.stack((X_submission[['temperature-1', 'mm_precip-1', 'ctx-1']].values,
                     X_submission[['temperature-2', 'mm_precip-2', 'ctx-2']].values,
                     X_submission[['temperature-3', 'mm_precip-3', 'ctx-3']].values,
                     X_submission[['temperature-4', 'mm_precip-4', 'ctx-4']].values),
                     axis = 1)
					 
					 
```

We construct a neural network by separating it into two parts, one for the time series data and one for the static data. The time series data is fed into an LSTM layer, the static data is fed into a dense layer

```python

import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Input, concatenate, Dropout, Bidirectional
from keras.regularizers import l1, l2


# Construct eh neural network by separating it into two parts, one for the time series data and one for the static data
# The time series data is fed into an LSTM layer, the static data is fed into a dense layer

ts_input = tf.keras.Input(shape=(4, 3), name='ts_input')
static_input = tf.keras.Input(shape = (17,), name='static_input')

####################################################################################################

LSTM1 = LSTM(64, activation='tanh', return_sequences=True)(ts_input)
dropout_lstm = Dropout(0.1)(LSTM1)
LSTM2 = LSTM(64, activation = 'tanh', return_sequences = False)(dropout_lstm)
lstm_out = Dropout(0.1)(LSTM2)

####################################################################################################

static1 = Dense(32, activation='relu')(static_input)
dropout_static = Dropout(0.1)(static1)
static2 = Dense(64, activation = 'relu')(dropout_static)
dropout_static2 = Dropout(0.1)(static2)
static3 = Dense(48, activation = 'relu')(dropout_static2)
dropout_static3 = Dropout(0.1)(static3)
static4 = Dense(32, activation = 'relu')(dropout_static3)
static_out = Dropout(0.1)(static4)

####################################################################################################

merged_out = concatenate([lstm_out, static_out]) # join the outputs of the LSTM and dense layers and pass them into a final neuron

layer_out1 = Dense(32, activation = 'relu')(merged_out)
dropout_out1 = Dropout(0.1)(layer_out1)
layer_out2 = Dense(24, activation = 'relu')(dropout_out1)
dropout_out2 = Dropout(0.1)(layer_out2)
layer_out3 = Dense(24, activation = 'relu')(dropout_out2)
dropout_out3 = Dropout(0.1)(layer_out3)
out_result = Dense(1, activation='sigmoid')(dropout_out3)

model = tf.keras.Model(inputs=[ts_input, static_input], outputs=out_result)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

```

#### Neural Network Structure

![Neural Network](/capstone-project/neural-network-structure.png)

#### Neural Network Results

![Neural Network Results](/capstone-project/neural-network-results.png)


We get recall similar results to that of the xgb model --> Further improvements should come from the hand of a better preproccesing probably, working with outliers and incorrect data in a more detailed way

## Heatmap and Buffer Analysis

Search for areas that are near existing bike lanes but do not have a nearby bike station. The buffer has been set to 500 meters.

### a. Libraries

```python

import folium
from folium.plugins import HeatMap
import pandas as pd
import json
import numpy as np
import dask.dataframe as dd
import dash
from dash.dependencies import Input, Output
from dash import dcc as dcc
from dash import html as html
import base64
import os
from flask import Flask
from io import BytesIO
from jupyter_dash import JupyterDash
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point, Polygon
import geopy.distance
import math

```

Create a joined dataframe that will be used for the heatmap analysis

```python
df_info1 = dd.concat([train_df, validation_df])
df_info.head()

```
![Heatmap Dataframe](/capstone-project/heatmap-dataframe.png)


### b. Dash Definition and HeatMap Generation

```python

# Dash definition
app = JupyterDash(__name__)

def generate_heatmap(year, month):
    selected_info = df_info[(df_info['year'] == float(year)) & (df_info['month'] == float(month))]
    print("Selected info:")
    print(selected_info.head())

    selected_info['average_percentage'] = selected_info[['percentage', 'ctx-4', 'ctx-3', 'ctx-2', 'ctx-1']].mean(axis=1)
    average_availability = selected_info.groupby('station_id').mean().reset_index()
    print(average_availability.head())

    heatmap_data = [(row['lat'], row['lon'], 1 - row['average_percentage']) for idx, row in average_availability.iterrows()]
    print(heatmap_data[:5])

    folium_map = folium.Map(location=[41.3870154, 2.1700471], zoom_start=13)



    with open('/content/drive/MyDrive/CapstoneProject/CARRIL_BICI.geojson') as f:
        bikelane_data = json.load(f)
    folium.GeoJson(bikelane_data).add_to(folium_map)

    HeatMap(heatmap_data).add_to(folium_map)

    # Map saved as html
    folium_map.save('/tmp/temp_map.html')

    # Copy in Google drive
    !cp /tmp/temp_map.html /content/drive/MyDrive/CapstoneProject/

# Initial map for the first year and month
generate_heatmap(years[0], months[0])

# Dash Layout
app.layout = html.Div([
    html.H1("Disponibilidad de estaciones de Bicing"),
    html.H2(id='map-title', children="Disponibilidad: Mes/Año"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in years],
        value=years[0]
    ),
    dcc.Dropdown(
        id='month-dropdown',
        options=[{'label': str(month), 'value': month} for month in months],
        value=months[0]
    ),
    html.Iframe(id = 'map', srcDoc = open('/content/drive/MyDrive/CapstoneProject/temp_map.html', 'r').read(), width = '100%', height = '600')
])

   # Map update
@app.callback(
    [Output('map', 'srcDoc'),
    Output('map-title', 'children')],
    [Input('year-dropdown', 'value'),
    Input('month-dropdown', 'value')])
def update_map(year, month):
    generate_heatmap(year, month)
    return open('/content/drive/MyDrive/CapstoneProject/temp_map.html', 'r').read(), f'Disponibilidad: {month}/{year}'

if __name__ == '__main__':

   app.run_server(debug=True)
   
   
from math import radians, cos, sin, asin, sqrt

def calculate_distance(lat1, lon1, lat2, lon2):

    # Convert coordinates from degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km

    return c * r * 1000  # Convert to meters
   
```

### c. Bicing HeatMap Generation

![Heatmap](/capstone-project/bicing-heatmap-capture.png)

![Heatmap2](/capstone-project/bicing-heatmap-capture2.png)
