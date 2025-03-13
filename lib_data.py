import pandas as pd
import requests
import json

import os

# Function to fetch spot prices from Energinet API
def fetch_spot_prices(start_date, end_date):
    # Read data from csv file if it exists
    try:
        df = pd.read_csv('data\spot_prices_test.csv')
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        return df
    except FileNotFoundError:
        pass

    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        'start': start_date.strftime('%Y-%m-%dT%H:%M'),
        'end': end_date.strftime('%Y-%m-%dT%H:%M'),
        'filter': json.dumps({"PriceArea": "DK1"})
    }
    response = requests.get(url, params=params)
    data = response.json()['records']
    df = pd.DataFrame(data)
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    return df

def fetch_gas_prices(start_date, end_date):
    # Read data from csv file if it exists
    try:
        df = pd.read_csv('data\gas_prices_test.csv')
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        return df
    except FileNotFoundError:
        pass

    url = "https://api.energidataservice.dk/dataset/GasDailyBalancingPrice"
    params = {
        'start': start_date.strftime('%Y-%m-%dT%H:%M'),
        'end': end_date.strftime('%Y-%m-%dT%H:%M'),
    }
    response = requests.get(url, params=params)
    data = response.json()['records']
    df = pd.DataFrame(data)
    #expand gas day to 24 hours
    df = df.loc[df.index.repeat(24)].reset_index(drop=True)
    #add hourdk column 
    df['HourDK'] = pd.date_range(start_date, periods=len(df), freq='h')
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    # only keep HourDk and EEXSpotIndexEUR_MWh	
    df = df[['HourDK', 'EEXSpotIndexEUR_MWh']]
    return df

def fetch_forecast_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,wind_speed_100m',
        'forecast_days': 7,
        'timezone': 'auto',
        'wind_speed_unit': 'ms'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract hourly data
    hours = pd.to_datetime(data['hourly']['time'])
    temp_2m = data['hourly']['temperature_2m']
    wind_speed_100m = data['hourly']['wind_speed_100m']

    forecast_df = pd.DataFrame({
        'HourDK': hours,
        'wind_speed': wind_speed_100m,
        'temperature': temp_2m
    })
    return forecast_df

# Function to fetch historical weather data from DMIs API
def fetch_data(url, start_date, end_date, parameter, station_id, api_key, historical=False):
    if historical:
        # Read data from csv file if it exists
        try:
            # if os is linux
            if os.name == 'posix':
                df = pd.read_csv(f'data/{parameter}_test.csv')
            else:
                df = pd.read_csv(f'data\{parameter}_test.csv')
            df['observed'] = pd.to_datetime(df['observed'])
            return df
        except FileNotFoundError:
            print("File not found: " +  f'data\{parameter}_test.csv')
            pass
    else:
        params = {
            'datetime': f'{start_date.strftime("%Y-%m-%dT%H:%M:%SZ")}/{end_date.strftime("%Y-%m-%dT%H:%M:%SZ")}',
            'parameterId': parameter,
            'stationId': station_id,
            'limit': 300000
        }
        headers = {'X-Gravitee-Api-Key': api_key}
        response = requests.get(url, params=params, headers=headers)
        data = response.json()['features']
        df = pd.DataFrame([obs['properties'] for obs in data])
        df['observed'] = pd.to_datetime(df['observed'])

        # Resample data to hourly frequency
        df.set_index('observed', inplace=True)
        numeric_columns = df.select_dtypes(include=['number']).columns
        df_numeric = df[numeric_columns]
        return df_numeric.resample('h').mean().reset_index()
    