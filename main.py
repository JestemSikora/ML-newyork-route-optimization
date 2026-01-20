# Frameworks that we're going to use
import pandas as pd
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split

# Functions
#from weather_api import download_weather_api

city = 'New York'

# History data
# Picking right coulmns for our problem
df = pd.read_parquet(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\yellow_tripdata_2025-01.parquet',
                     columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'congestion_surcharge',
                              'PULocationID', 'DOLocationID'])



df['user_id'] = np.arange(len(df))
df.set_index('user_id', inplace=True)
df.sort_index(inplace=True)

# Renaming to know units
df = df.rename(columns={'trip_distance': 'trip_distance [km]'})

# Feature engineering
df['time_diffrence'] =  df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'] 

# Second table containing names of all places
df_dist = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\id_lookup.csv')
df_dist_OSM = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\OSM_Street_lookup.csv', delimiter=';')


# Merging tables on location id
df = pd.merge(df, df_dist, left_on='PULocationID', right_on='LocationID')
df = pd.merge(df, df_dist, left_on='DOLocationID', right_on='LocationID')
df = pd.merge(df, df_dist_OSM, left_on='Zone_x', right_on='NTA')
df = pd.merge(df, df_dist_OSM, left_on='Zone_y', right_on='NTA')


# Filtering important columns
df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'time_diffrence','trip_distance [km]', 'RatecodeID', 'congestion_surcharge',
                'PULocationID', 'OpenStreetMap_x', 'Borough_x', 'DOLocationID', 'OpenStreetMap_y', 'Borough_y']]

# Renaming for better convenience
df = df.rename(columns={'OpenStreetMap_x': 'PULZone'})
df = df.rename(columns={'Borough_x': 'PULBorough'})
df = df.rename(columns={'OpenStreetMap_y': 'DOLZone'})
df = df.rename(columns={'Borough_y': 'DOLBorough'})


# Changing timedelta64[us] output to Hours
df['time_diffrence'] = df['time_diffrence'].dt.total_seconds() / 3600
df = df.rename(columns={'time_diffrence': 'time_diffrence [h]'})

# Average speed 
df['average_speed [km/h]'] = round(df['trip_distance [km]'] / df['time_diffrence [h]'],2)

# Rounding pickup time to 1 hour for api weather data
df['tpep_pickup_datetime'].round('1h')

# Reading weather csv & changing datatype to datetime64[us]
df_weather = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\weather-data.csv')
df_weather['Time'] = pd.to_datetime(df_weather['Time']).astype('datetime64[us]')

# Merging df (taxi data) and df_weather (weather data)
df_time_weather = pd.merge(df, df_weather, left_on='tpep_pickup_datetime', right_on='Time')


'''
print(f'DataFrame df_time_weather: {df_time_weather.columns}')
print(f'DataFrame df kolumny: {df.columns}') '''

# Dropping useless columns
df_time_weather = df_time_weather[['Time', 'Temperature', 'Snowfall',
       'Showers', 'Rain', 'Visibility', 'Precipitation', 'Wind_speed_10m']]

# Final changes in df
df = df.join(df_time_weather, lsuffix='_taxi', rsuffix='_weather')
df = df.drop(columns=['Time'])
df = df.sample(frac=1).reset_index(drop=True)

# Changing datatypes to 'category' and numbers for XGBoost
# Category
cat_cols = ['PULZone', 'PULBorough', 'DOLZone', 'DOLBorough']
for i in cat_cols:
    df[i] = df[i].astype('category')

# Numbers
df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
df['dropoff_hour'] = pd.to_datetime(df['tpep_dropoff_datetime']).dt.hour
df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

# Checking data types
print(df.dtypes)

# Saving final dataset to csv
df.to_parquet('dataset-marged.parquet')

