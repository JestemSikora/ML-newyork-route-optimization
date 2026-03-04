# Frameworks
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import json

def merching_data(df):
    '''
    Intagrates raw trip data with lookups and weather sources.

    '''
    # History data
    # Picking right coulmns for our problem
    df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'RatecodeID', 'congestion_surcharge',
            'PULocationID', 'DOLocationID', 'fare_amount', 'extra', 'tolls_amount']]

    # Renaming to know units
    df = df.rename(columns={'trip_distance': 'trip_distance km'})

    # Feature engineering
    df['time_diffrence'] =  df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'] 


    # Second table containing names of all places
    df_dist = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\id_lookup.csv')
    df_dist_OSM = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\OSM_Street_lookup.csv', delimiter=';')


    # Merging tables on location id
    df = pd.merge(df, df_dist, left_on='PULocationID', right_on='LocationID', how='left')
    df = pd.merge(df, df_dist, left_on='DOLocationID', right_on='LocationID', how='left')


    # Filtering important columns
    df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'time_diffrence','trip_distance km', 'RatecodeID', 'congestion_surcharge',
                    'PULocationID', 'Borough_x', 'DOLocationID', 'Borough_y', 'fare_amount', 'extra',
                                    'tolls_amount']]

    # Renaming for better convenience
    df = df.rename(columns={'Borough_x': 'PULBorough'})
    df = df.rename(columns={'Borough_y': 'DOLBorough'})


    # Changing timedelta64[us] output to Hours
    df['time_diffrence'] = df['time_diffrence'].dt.total_seconds() / 3600
    df = df.rename(columns={'time_diffrence': 'time_diffrence h'})


    # Average speed 
    df['average_speed km/h'] = round(df['trip_distance km'] / df['time_diffrence h'],2)
    df['average_speed km/h'] = df['average_speed km/h'].replace(0, np.nan)
    df['average_speed km/h'] = df['average_speed km/h'].replace([np.inf, -np.inf], np.nan)


    # Saving old datetimes for better model understanding
    df['orginal_pickup_datetime'] = df['tpep_pickup_datetime']
    df['orginal_dropoff_datetime'] = df['tpep_dropoff_datetime']

    # Rounding pickup time to 1 hour for api weather data
    df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].dt.round('h')
    df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].dt.round('h')


    # Reading weather csv & changing datatype to datetime64[us]
    df_weather = pd.read_csv(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\weather-data.csv')
    df_weather['Time'] = pd.to_datetime(df_weather['Time']).astype('datetime64[us]')


    # Setting index to join two datasets
    df_time_weather = df_weather.set_index('Time', drop=True)

    # Marching to one dataset
    df = pd.merge(df, df_weather, left_on='tpep_pickup_datetime', right_on='Time', how='left')
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop(columns=['Visibility'])

    # Clear all NaN
    df = df.dropna()

    # Changing datatypes to 'category' and numbers for XGBoost
    # Category
    cat_cols = ['PULBorough','DOLBorough']
    for i in cat_cols:
        df[i] = df[i].astype('category')

    # Changing dtypes to int and dropping columns
    df = df.rename(columns={'orginal_pickup_datetime': 'pickup_', 'orginal_dropoff_datetime': 'dropoff_'})
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'Time'])
    print(f'DataFrame after merching: {df.shape}')
    return df

def cleaning_data(df):
    '''
    Removes system recording errors and outliers.
    '''
    df_negative_fare = df['fare_amount'] < 0 
    df_negative_fare = df_negative_fare.loc[df_negative_fare]
    df = df.drop(df_negative_fare.index)


    df['user_id'] = np.arange(len(df))
    df.set_index('user_id', inplace=True)
    df.sort_index(inplace=True)

    # Deleting time_diffrence outliers
    time_mean = df['time_diffrence h'].mean()
    time_std = df['time_diffrence h'].std()

    upper_limit = time_mean + 3*time_std
    lower_limit = time_mean - 3*time_std

    upper_limit_df = df[df['time_diffrence h'] > upper_limit].index
    lower_limit_df = df[df['time_diffrence h'] < lower_limit].index

    df = df.drop(upper_limit_df)
    df = df.drop(lower_limit_df)


    # Deleting errors in difference - travel time can't be negative
    df['time_diffrence h'].astype(np.int64)
    df_negative_travel_time  = (df['time_diffrence h'] < 0)
    df_negative_travel_time = df_negative_travel_time.loc[df_negative_travel_time]
    df = df.drop(df_negative_travel_time.index)


    # Congestion surcgarge error's
    df_negative_cong = (df['congestion_surcharge'] < 0)
    df_negative_cong = df_negative_cong.loc[df_negative_cong]
    df = df.drop(df_negative_cong.index)
    print(f'DataFrame after clearing: {df.shape}')

    return df

def feature_engineering(df):
    '''
    Generates new predictive features.
    '''
    cols = ['pickup_', 'dropoff_']
    for i in cols:
        df[i+'hour'] = (df[i].dt.hour).astype('float32')
        df[i+'min'] = (df[i].dt.minute).astype('float32')
        df[i+'month'] = (df[i].dt.month).astype('float32')
        df[i+'day'] = (df[i].dt.day).astype('float32')

        # Taking minutes and hours to change it into one feature:
        # How many minutes has passed since the 00:00?

        df[i+'minutes_after_midnight'] = df[i+'min'] + df[i+'hour'] * 60

    df = df.drop(columns=['pickup_', 'dropoff_', 'pickup_hour', 'pickup_min', 'dropoff_hour', 'dropoff_min'])
    print(f'DataFrame after feature engineering: {df.shape}')
    return df

def pruning(df):
    '''
    Balances the dataset by downsampling to prevent model bias
    and optymalize memory usage.
    '''
    df_before = df.copy()
    # Creating 'route' column to group the same routes
    df['min_id'] = np.where(df['PULocationID'] <= df['DOLocationID'], df['PULocationID'], df['DOLocationID'])
    df['max_id'] = np.where(df['PULocationID'] > df['DOLocationID'], df['PULocationID'], df['DOLocationID'])
    df['route'] = df['min_id'].astype(int).astype(str) + ' - ' + df['max_id'].astype(int).astype(str)

    # JSON
    with open(r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\notebooks\most_common_routes_list.txt', 'r', encoding='utf-8') as file:
        most_common = json.load(file)

    # JSON has two dictionaries
    route_names_dict = most_common['route']  
    counts_dict = most_common['count']       

    # Save names of indexe's that meet the condition
    routes_to_sample = []
    for idx, count in counts_dict.items():
        if count > 500:
            route_name = route_names_dict[idx]
            routes_to_sample.append(route_name)

    # Change to set for speed
    routes_to_sample = set(routes_to_sample)

    # Filtering
    mask = df['route'].isin(routes_to_sample)
    df_keep = df[~mask]
    df_to_sample = df[mask]

    if not df_to_sample.empty:
        # Function checks: take 1000 random row or actual value
        # Using: Split-Apply-Combine method to replace 'for' function
        df_sampled = df_to_sample.groupby('route', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 500), random_state=123),
            include_groups=False).reset_index(drop=True)
        
        # Concat outputs
        df = pd.concat([df_keep, df_sampled]).reset_index(drop=True)
        print(f"Number of rows deleted: {len(df_before) - len(df)}")

    print(f'Before: {df_before.shape}')
    print(f'After: {df.shape}')

    return df

def preprocess_data(df):
    '''
    The main orchestrator of the data transformation pipeline.
    '''
    df = df.copy()

    df = merching_data(df)
    df = cleaning_data(df)
    df = feature_engineering(df)
    df = pruning(df)

    df = df.drop(columns=['route', 'min_id', 'max_id'])

    return df

def main_process():
    '''
    Executes batch processing and consolidation of the NYC taxi dataset.
    '''
    # use pathlib for inteligent classification what slash to use base on OS
    folder_path = Path(r"C:\Users\wikto\OneDrive\Dokumenty\all-datasets\taxi_routes_datasets_23-25")

    # check if the files exist in the target path
    if not folder_path.exists():
        print('Error! Folder path does not exist.')

    list_of_files = list(folder_path.glob("*.parquet"))

    if not list_of_files:
        print('Error! No *.paraquet files in the folder')
    preprocessed_data_list = []

    for file in list_of_files:
        print(f'Preprocessing: {file.name}')
        
        df = pd.read_parquet(file)
        print(f'Orginal shape before preprocessing: {df.shape}')
        
        df_preprocessed = preprocess_data(df)

        preprocessed_data_list.append(df_preprocessed)
        print('-'*30)

    print('Final dataset merching...')
    df = pd.concat(preprocessed_data_list, ignore_index=True)

    df.to_parquet(path=r'C:\Users\wikto\OneDrive\Dokumenty\AA_projects\road-optimization\data\preprocessed-dataset')
    print("Data preprocessing ended successfully.")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    main_process()
