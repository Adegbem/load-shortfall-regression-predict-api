"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # creating a function that can split the time column
    def convert_time(row):
        date, time = row.split(' ')
        year, month, day = date.split('-')
        hour = time.split(':')[0]
        return year, month, day, hour  # we can also return a pd.Series([...]) and not use a zip function later on

    # splitting the time column into features
    feature_vector_df['year'], feature_vector_df['month'], feature_vector_df['day'], feature_vector_df['hour'] \
        = zip(*feature_vector_df['time'].map(convert_time))

    # we need to convert the new features to numeric and drop the old time column
    cols = ['year', 'month', 'day', 'hour']
    feature_vector_df[cols] = feature_vector_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    feature_vector_df.drop('time', axis=1, inplace=True)

    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)').astype('int64')
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract('(\d+)').astype('int64')
    feature_vector_df = feature_vector_df.drop(['Unnamed: 0'], axis=1)

    def classify_parameters(input_df):
        weather_dict = {}
        for x in input_df.columns:
            finder = x.find('_')
            y = x[finder + 1:]
            weather_dict[y] = weather_dict[y] + ',' + x if y in weather_dict else x
        return weather_dict

    # getting the weather parameters dictionary
    weath_param = classify_parameters(feature_vector_df)

    # getting name of all temperature variables alike
    temp_max, temp, temp_min = weath_param['temp_max'].split(','), weath_param['temp'].split(','), \
                               weath_param['temp_min'].split(',')
    # temp_merge = [temp_max, temp, temp_min]
    temp_list = temp_max + temp + temp_min

    # Identifying column to drop from our dataset
    usable_high_corr = ['Valencia_temp_min', 'Madrid_temp', 'Madrid_temp_max']
    drop_coln_list = [x for x in temp_list if x not in usable_high_corr]
    # Adding humidity and pressure columns to be dropped
    final_drop_list = drop_coln_list + ['Madrid_humidity', 'Valencia_pressure']

    # we will be dropping for both the train and the test dataset
    feature_vector_df = feature_vector_df.drop(final_drop_list, axis=1)

    # Creating a function that combines the various location by the classify weather parameters
    # Combination is done by using there mean value
    def impute(input_df):
        input_df = input_df.copy()
        weather_dict = classify_parameters(input_df)
        for x, y in weather_dict.items():
            coln = y.split(',')
            if len(coln) < 2:
                continue
            else:
                input_df[x] = input_df[coln].mean(axis=1)
        return input_df

    predict_vector = impute(feature_vector_df)

    # renaming the temperature(s) columns
    predict_vector.rename(columns = {'Valencia_temp_min':'temp_min', 'Madrid_temp_max':'temp_max'
        ,'Madrid_temp':'temp'}, inplace = True)
    # Creating new features temperature range
    predict_vector['temp_range'] = predict_vector['temp_min'] - predict_vector['temp_max']
    # Creating new features wind_force
    predict_vector['wind_force'] = predict_vector['wind_speed'] * predict_vector['wind_deg']
    # removing unuseful features from our model (was informed using "Features_Importance")
    unuseful = ['Bilbao_wind_speed', 'Bilbao_wind_deg', 'Valencia_wind_speed', 'Barcelona_pressure',
                'Valencia_wind_deg', 'Madrid_pressure', 'pressure', 'Bilbao_weather_id',
                'Barcelona_weather_id', 'Bilbao_rain_1h', 'Seville_clouds_all', 'Barcelona_wind_speed',
                'Barcelona_rain_1h', 'Barcelona_rain_3h', 'Bilbao_snow_3h', 'Seville_rain_3h', 'rain_3h',
                'Valencia_snow_3h', 'year', 'snow_3h', 'Seville_rain_1h', 'Barcelona_wind_deg','rain_1h',
                'Madrid_rain_1h', 'clouds_all', 'Bilbao_clouds_all', 'Madrid_weather_id', 'wind_deg',
                'Seville_weather_id', 'Seville_wind_speed', 'weather_id', 'Madrid_clouds_all',
                'Seville_pressure', 'temp_range', 'humidity', 'Seville_humidity']
    predict_vector = predict_vector.drop(unuseful, axis=1)
    # ------------------------------------------------------------------------

    return predict_vector


def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""


def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
