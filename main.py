# Frameworks that we're going to use
import pandas as pd
import osmnx as ox

# Functions
#from weather_api import download_weather_api

city = 'New York'

# History data
# Picking right coulmns for our problem
df = pd.read_parquet(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\yellow_tripdata_2025-01.parquet',
                     columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'congestion_surcharge',
                              'PULocationID', 'DOLocationID'])

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
#print(df[['average_speed [V]', 'trip_distance [km]', 'time_diffrence [h]']].head())

# Changing time for 1h frequency
df_time_for_api = df['tpep_pickup_datetime'].copy()
df_time_for_api = df_time_for_api.dt.round('1h')



df_weather = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\weather-data.csv')
#print(df_time_for_api.dtypes, df_weather.dtypes)

df_weather['Time'] = pd.to_datetime(df_weather['Time']).astype('datetime64[us]')

df_time_weather = pd.merge(df_time_for_api, df_weather, left_on='tpep_pickup_datetime', right_on='Time')
df_time_weather = df_time_weather.drop(columns=['Time'])


print(df_time_weather)

