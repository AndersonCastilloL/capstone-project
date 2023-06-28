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

The Bicing service is a great way to navigate through the city of Barcelona, leveraging its extensive bike lane network. Nowadays, electric bikes are also available, and so the usage of the service has grown quite a bit over the years.
In this study the main objective is to try to predict the availability of free docks throughout the city during march of 2023 using data from the previous 4 years, this being a really important feature to guide the users to the best stations, not westing any time looking for bikes/free docks.
Apart from this prediction task, we have also looked at the evolution of the usage of the network by using heatmaps, and tried to look for places along bike lanes where the construction of future stations could make sense.

The notebook where all the analysis has been done can be found in the GitHub repository, but due to the huge size of the datasets, the full project can be found in the following [Google Drive](https://drive.google.com/drive/folders/1ZIY2ZMhsCITuSFC1bnIDJrbZh-sS63HP?usp=drive_link) folder. In there you will be able to find all the data used, aswell as the models and other figures and material.

## Gathering the Data

For this data project, we have started off with the following data:

- The bicing stations information and status, obtained from [Open Data BCN](https://opendata-ajuntament.barcelona.cat)

- The weather of the city of Barcelona, obtained from the [Meteo Cat](https://www.meteo.cat) public API


The following script was used to download the data from [Open Data BCN](https://opendata-ajuntament.barcelona.cat) by year and month:

```bash
import os

i2m = list(zip(range(1,13), ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']))
for year in [2022, 2021, 2020, 2019]:
    for month, month_name in i2m:        
        os.system(f"wget 'https://opendata-ajuntament.barcelona.cat/resources/bcn/BicingBCN/{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")
        os.system(f"7z x '{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")
        os.system(f"rm '{year}_{month:02d}_{month_name}_BicingNou_ESTACIONS.7z'")

```

And the script used to do requests on the [Meteo Cat](https://www.meteo.cat) public API can be found on the Google Drive of the project, in the weather folder.

### b. Data merging

The data obtained from [Open Data BCN](https://opendata-ajuntament.barcelona.cat) is separated into multiple datasets, each one containing either the information or the status of the stations for each month. To work with so many huge datasets at the same time, the processing of the data was done in Tableau using Tableau Prep which give us more flexibility to join, filter and build the next structure according to our metadata-sample-submission.csv

#### Metadata Sample Submission

![Metadata Sample Submission](/capstone-project/metadata-sample-submission.png)

## Data exploration

All of the following analysis, as already mentiones, has been done with tableau due to its low Ram demand and speed.

### a. Tableau Workflow

The first flow in Tableau Prep concatenates all the bicing station status files of all years and creates new columns like year, month, day and hour. 
The flow has two sub-flows because is necessary to also apply the same process to bicing station 2023 files. Both flows create new files with a hyper format which is easier to manage and control.

![First Tableau flow](/capstone-project/first-tableau-flow.png)

The second flow in Tableau Prep gets the hyper files of bicing station status (2019-2022) and merges them with the last bicing station information (March, 2023) to get information on longitude, latitude, name, capacity and postcode.
At this point the first relevant approximation is made, as the capacity of the stations in the training data is taken as the one from 2023 and not the "real" one.

![Second Tableau flow](/capstone-project/second-tableau-flow.png)

The same process is applied to the bicing station status 2023 but filtering docks_availability and bike_availability. These fields will be evaluated and predicted by the models.

![Third Tableau flow](/capstone-project/third-tableau-flow.png)

Finally, the last flow transforms the input data of the preview flow using an aggregation to year, month, day and hour, calculating averages on the rest of fields and creating four additional fields with the percentage of docks available in the four hours before.

![Fourth Tableau flow](/capstone-project/fourth-tableau-flow.png)

Before training and testing, the data is analyzed to identify patterns, outliers and to visualize relevant features.

### b. Visual Analysis in Tableau

Before training and testing with machine learning models, we used the following charts to visually analyze the data constructed based on the final output to identify outliers, COVID behaviors and relevant characteristics 

![Bicing Data Analysis](/capstone-project/bicing-data-analysis.png)

Where we can see that as expected, the percentage of docks available increases as we get to upper areas of the city, where the users take the bikes and ride them downhill to the lower areas.
Also, we canb observe the effects of covid, especially in 2020 where the lockdown shows up as no available bikes.

## Prepare the data for Machine Learning algorithms

Once we have merged all the datasets into one huge one, the next step is to process all the information, now in python using Google Colab. All of this process has been done in the following steps:

### b. Data Cleaning and Filtering

To start, we have to clean the raw data that Tableau outputs following:

- Initial cleaning to go from the format that the Tableau merger outputs to the one desired for the model. The objective of this first step is to leave the training data in the same format as the submission one.
- Drop all rows with out of service stations and year 2020, to avoid COVID effects on our training data.
- Select the features with the highest relevance (station_id, lat, lon, year, month, day, hour, ctx-4, ctx-3, ctx-2, ctx-1, percentage).

- Split the data, by leaving 1029, 2021 and 2022 for training and 2023 for validation.

- Filter only the data with status "IN_SERVICE".

### c. Processing pipelines

Apart from the preprocessing done using tableau to create the initial dataframe, everything else will be done using sklearn pipelines. We have built 4 different ones:

- train_preparator: Adds the weather data, classifies the registers in 5 hour intervals and marks them with a 1 if they correspond to a weekend day, selects only 5 hours of the day to prevent overlapping between registers and normalizes the time data. 
- val_preparator: Same procedure as train_preparator but selecting all 24 hours of the day to resemble what the submission will contain
- submission_preparator: Again, same procedure, but also adding the location of the station to the dataset, as the one that we are using, downloaded from kaggle does not contain this valuable informatio
  
- scaling_pipeline: 
  - to fill and scale certain columns in the data, fitting it to the training data, and transforming both training and submission data.


### d. Classes and Functions

Each pipeline is supported by a set of functions and classes to structure the data and features. The functions are applied using classes that can be inserted in the pipelines and are the following:

- hour_selector: filter a range of hours to avoid overlapping with the columns with the same information of one to four hours before
  
- extra_time_info: add new columns of time like is_weekend, timeframe1, timeframe2, timeframe3, timeframe4, timeframe5 to group the hours. These timeframe columns are used to mitigate the effects of only training with 5 hours, by grouping all the ones in between.

- time_norm: Time normalizer function to normalize the time columns to a -1/1 scale with periodicity. This is done by applying a sin and cos transformation to the columns
  
- weather_merge: Merge the main dataframe with weather dataframe and split datetime column in year, month, day and hour



### e. Prepare all data and load data that has been saved
Once we have our pipelines ready we pass through them the data to obtain 4 datasets.

We have three datasets. The first dataset is for training purposes, the second dataset is for validation, considering only a specific range of hours just like in the training one, the third one is another validation dataset but now containing all 24 hours to compare the effects of this restriction on the training process, and the fourth one is the submission dataset.


- Training Dataset

  ![X-Train dataset](/capstone-project/x-train.png)

- Validation Dataset (5 hr)

  ![X-Val dataset](/capstone-project/x-val.png)

- Validation Dataset with all hours (24 hr)

  ![X-Val24 dataset](/capstone-project/x-val24.png)

- Submission Dataset (Testing Dataset)

  ![X-Submission dataset](/capstone-project/x-submission.png)

## Model Selection and Training

For this project we will use 2 different models. The first one is an xgboost regressor, that has been chosen following other similar projects to this one and that has outperformed with less training times other similar ensemble models. Then, a neural network containing LSTM layers is used to capture the temporal features and their ordering.
We will start off with the xgboost models.

### a. XGBregressor

We will first use a data subset to perform a grid search on the parameter space to find the ones that fit the best our xgboost model.
 
#### Parameter tuning using GridSearchCV

We apply grid searhc with cross validation find the best parameters for our model after transforming our data using pipelines. We use an XGBRegressor and a set of parameters associated with it to evaluate the optimal parameters for training and prediction.

```python
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

#### Training the XGBRegressor model

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

##### Loss Function Graphic

![Loss Function graphic](/capstone-project/loss-function-xgboost.png)

##### Most Relevant Features

![Feature Importances Graphic](/capstone-project/feature-importance-xgboost.png)

The feature important plot is really interesting. From it we can conclude that:

- A expected, the previous hours percentage gives a lot of info about how many docks will be available in the following hours.
- The location of the station are the next most relevant features. The usage of the service is not homogeneous across the city and this impacts to availability of docks
- timeframe2, corresponding to peak hours in the morning has quite a bit of weight. The stations network probably has the most activity in these hours
- the hour also affects quite a bit the final %, as expected --> Since we are only training with
- The weather does not even appear in the graph. We expected way more weight from those variables, especially rain. We guess that since Barcelona is not a very rainy city, the model does not see many registers with rain and so it end up ignoring completely the variable.

#### Second XGBRegresor model

In earlier version of the code we worked with a more complex model than the one that GridSearch found to be the best, and ended up getting better results. To try to replicate that behaviour and see if we can get a better performance we have also trained the following model using the same procedure:

```python

xgb_model2 = XGBRegressor(objective='reg:squarederror',
                          n_estimators=2000,
                          max_depth=10,
                          learning_rate=0.1,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          gamma=0.1,
                          reg_alpha=0.1,
                          reg_lambda=10.0,
                          n_jobs=-1,
                          random_state=128,
                          early_stopping_rounds = 30)

```
Obtaining a slightly lower final rmse score (0.1019 vs 0.1100) and the following dfeature imporances and loss evolution curves:

![Loss Function graphic](/capstone-project/xgb_loss2.png)
![Feature Importances Graphic](/capstone-project/xgb2_features.png)

Which in this case show quite a bit more overfitting due to the complexity of the model and also more relevance given to the location of the stations. The temperature now also appears in the feature importance plot, but still not the rain.

### b. LSTM

Apart from our xgbregressor model, and since our data contains historical data, we have also tried to study the prediction problem as a time-series analysis by working with long short-term memory layers on a neural network to try and exploit the temporal ordering of the information. Since not all the variables are historical, that have built a time series with just the one that are historical context, and we have treated the rest with a fully connected network. By using this approach our intention is to capture not only the static context of the data, but also the past that leads to each station's state.

To operate with the two groups of data, the nework starts off with 2 LSTM layers and a dense network that work in parallel and that process the temporal data and static data, respectively. After this, the outputs are concatenated and passed thorugh 3 final dense layers that lead to a single neuron with a sigmoid activation function that returns a value between 0 and 1.
We have also added droupout layers to prevent overfitting (we will see that that is a problem with this model)


#### Neural Network Structure

![Neural Network](/capstone-project/neural-network-structure.png)

![Neural Network](/capstone-project/lstm_structure.png)

#### Neural Network Results

![Neural Network Results](/capstone-project/lstm_loss.png)

#### Neural Network Feature Importance

![Neural Network Feature Importance](/capstone-project/lstm_features.png)


We get similar results to that of the xgb model, which makes us feel like further improvements should come from the hand of a better preproccesing probably, working with outliers and incorrect data in a more detailed way. On the other hand, we see that in this case the weather variables have a lot more weight than in the xgb model, which is what we expected to happen in the first place, but the first two models didn't show.

## Heatmap and Buffer Analysis

This last section consists in two parts:

- Creation of a dinamic heat map of dock availability that can be changed based on the month and year parameter. This will allow us to understand the pattern of bicycle usage across the city.

- Performance of a buffer analysis. Here we are going to search for areas that are near the bike lanes but do not have a nearby bike station to detect potential zones where new bike stations could be installed. The buffer has been set in 500 meters based on the size of the city and the number of stations.


### a. Heatmap

The code is designed to generate and display the dinamic heatmap that represents the average availability of bike docks across the city, given a specific month and year.First, lets create a joined dataframe that will be used for the heatmap analysis


![Heatmap Dataframe](/capstone-project/heatmap-dataframe.png)

Now we are going to calculate the average percentage of bike availability at each station using the historical data. then we will generate a heatmap based on this data, with coordinates of each station and a color intensity indicating the level of bike availability. This heatmap is overlaid onto the city map created using Folium, providing a clear visualization of bike availability in different areas of the city.

The resulting map is saved as an HTML file in a temporary location and then the interface powered by Dash includes dropdown menus for selecting the month and year of interest. The map updates dynamically based on these selections, providing a tool for exploring dock availability across different times.

![Heatmap](/capstone-project/bicing-heatmap-capture.png)

![Heatmap2](/capstone-project/bicing-heatmap-capture2.png)


Remarks: If we compare the different periods, we can see that in general, the downtown is the area where there is a major use of bicycle and therefore less availability.

### b. Buffer Analysis

This produces a heatmap representing the average availability of bikes at different stations in march 2023 including the buffer analysis around bike lanes.

First, we define a function function to calculate the geographical distance between two points on Earth given their latitude and longitude. This function will be later used in the buffer analysis.

Next, we compute the average availability of bikes at each station for march 2023. This data is used to create a list of tuples where each tuple contains the latitude, longitude and availability of a bike station. The bike availability is subtracted from 1 to display lack of availability in the heatmap.

A base map of the city is created using Folium and the bike lanes data is loaded from a GeoJSON file and added to the map.The heatmap is added to the map using the previously computed bike station data.

The script then performs the buffer analysis. It iterates through each bike lane segment and checks if it's more than 500 meters away from any bike station. If it is, a marker is added to that point on the map. This analysis helps identify undeserved areas where additional bike stations might be needed. Then is saved as an html.

The output is the following:

[Buffer Analysis](/capstone-project/temp_map_no_mark23.html)

Remarks: With this buffer analysis we can see that there are several areas in the city that might need a station, such as Zona franca, the port of Barcelona or near Sant Adria de Besos.

