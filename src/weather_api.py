import requests
import pandas as pd

# New York lat&len
lat = 40.7128
lon = -74.0060

def download_weather_api(miasto, data_iso, data_end, lat, lon):
    """
    miasto: str 
    data_iso: str format "RRRR-MM-DD"
    data_end: str format "RRRR-MM-DD"

    """

    # Historical Data
    weather_url = "https://archive-api.open-meteo.com/v1/archive"

    # Defined params
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": data_iso,
        "end_date": data_end,
        "hourly": "temperature_2m,snowfall,showers,rain,visibility,precipitation,wind_speed_10m",
        "timezone": "auto"
    }

    # GET request to Open-Mateo
    response = requests.get(weather_url, params=params).json()

    # Variables from GET
    temp = response["hourly"]["temperature_2m"]
    czas = response["hourly"]["time"]
    snowfall = response["hourly"]["snowfall"]
    showers = response["hourly"]["showers"]
    rain = response["hourly"]["rain"]
    visibility = response["hourly"]["visibility"]
    precipitation = response["hourly"]["precipitation"]
    wind_speed_10m = response["hourly"]["wind_speed_10m"]


    # Data for pandas DataFrame
    data = {
        'Time': czas,
        'Temperature': temp,
        'Snowfall': snowfall,
        'Showers': showers,
        'Rain': rain,
        'Visibility': visibility,
        'Precipitation': precipitation,
        'Wind_speed_10m': wind_speed_10m
    }

    return data 
    

download = download_weather_api("New York", "2025-01-01", "2025-01-31", lat, lon)
df = pd.DataFrame(download)

#df.to_csv('weather-data.csv', index=False)

print(df)



