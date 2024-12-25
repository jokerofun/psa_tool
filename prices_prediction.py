import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

API_KEY = 'YOUR_DMI_API_KEY'
STATION_ID = '06030'  # Aalborg weather station

# DMI Data Retrieval Function
def fetch_data(url, start_date, end_date, parameter):
    params = {
        'datetime': f'{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}',
        'parameterId': parameter,
        'stationId': STATION_ID
    }
    headers = {'X-Gravitee-Api-Key': API_KEY}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()['features']
    df = pd.DataFrame([obs['properties'] for obs in data])
    df['observed'] = pd.to_datetime(df['observed'])

    # Resample data to hourly frequency
    df.set_index('observed', inplace=True)
    numeric_columns = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_columns]
    return df_numeric.resample('h').mean().reset_index()

# Spot Price Data Retrieval Function
def fetch_spot_prices(start_date, end_date):
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        'start': start_date.strftime('%Y-%m-%dT%H:%M'),
        'end': end_date.strftime('%Y-%m-%dT%H:%M'),
        'filter': json.dumps({"PriceArea": "DK1"}),
        'limit': 10000
    }
    response = requests.get(url, params=params)
    data = response.json()['records']
    df = pd.DataFrame(data)
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    return df

# Feature Engineering
def preprocess_data(df, wind_data, temp_data):
    df['hour'] = df['HourDK'].dt.hour
    df['weekday'] = df['HourDK'].dt.weekday
    df['PriceEUR'] = df['SpotPriceEUR']

    # Ensure both datetime columns are in the same timezone
    df['HourDK'] = pd.to_datetime(df['HourDK']).dt.tz_localize(None)
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    temp_data['observed'] = pd.to_datetime(temp_data['observed']).dt.tz_localize(None)

    wind_data.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    temp_data.rename(columns={'observed': 'HourDK', 'value': 'temperature'}, inplace=True)

    # Merge wind data with electricity prices
    df = pd.merge_asof(df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
    df = pd.merge_asof(df.sort_values('HourDK'), temp_data.sort_values('HourDK'), on='HourDK')
    df.dropna(inplace=True)
    return df

# Train-Test Split
def split_data(df):
    features = ['hour', 'weekday', 'wind_speed', 'temperature']
    target = 'PriceEUR'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Model Training and Prediction
def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.2f} EUR")
    # Plotting Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.legend()
    plt.title('Actual vs Predicted Spot Prices (DK1)')
    plt.xlabel('Test Samples')
    plt.ylabel('Price (EUR)')
    plt.show()

# Predict Future Prices (Next 7 Days)
def predict_future(model, asof):
    future_dates = [asof + timedelta(hours=i) for i in range(24*7)]
    future_df = pd.DataFrame({
        'hour': [d.hour for d in future_dates],
        'weekday': [d.weekday() for d in future_dates]
    })

    start_date = asof - timedelta(hours=1)
    end_date = start_date + timedelta(days=7)
    wind_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, end_date, 'wind_speed')
    temp_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, end_date, 'temp_dry')

    future_df['HourDK'] = future_dates

    # Ensure both datetime columns are in the same timezone
    future_df['HourDK'] = pd.to_datetime(future_df['HourDK']).dt.tz_localize(None)
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    temp_data['observed'] = pd.to_datetime(temp_data['observed']).dt.tz_localize(None)
    
    wind_data.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    temp_data.rename(columns={'observed': 'HourDK', 'value': 'temperature'}, inplace=True)

    # Merge wind data with future dates
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), temp_data.sort_values('HourDK'), on='HourDK')
    future_df.dropna(inplace=True)

    # Drop the 'HourDK' column before making predictions
    future_df = future_df.drop(columns=['HourDK'])

    future_predictions = model.predict(future_df)
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green')
    plt.title('Predicted Spot Prices for the Next 7 Days (DK1)')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.legend()
    plt.show()
    return future_df, future_predictions

# Main Execution
if __name__ == "__main__":
    asof = datetime(2024, 12, 1)
    start_date = datetime(2020, 1, 1)

    df = fetch_spot_prices(start_date, asof)
    wind_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'wind_speed')
    temp_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'temp_dry')

    df = preprocess_data(df, wind_data, temp_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    future_df, future_predictions = predict_future(model, asof)

    # Attach hours to future predictions
    future_df['PriceEUR'] = future_predictions
    future_df['HourDK'] = asof
    future_df['HourDK'] = future_df['HourDK'] + pd.to_timedelta(future_df['hour'], unit='h')
    future_df.set_index('HourDK', inplace=True)
    future_df.drop(columns=['hour'], inplace=True)

    print(future_df)