import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Data Retrieval Function (Example: DK1 Spot Prices API)
def fetch_data(start_date, end_date):
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

# Fetch Wind Speed Data from DMI
def fetch_wind_data(start_date, end_date):
    url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
    params = {
        'datetime': '{}/{}'.format(start_date.strftime('%Y-%m-%dT%H:%M:%SZ'), end_date.strftime('%Y-%m-%dT%H:%M:%SZ')),
        'parameterId': 'wind_speed',
        'stationId': '06030' # Aalborg weather station
    }
    headers = {
        'X-Gravitee-Api-Key': 'YOUR_DMI_API_KEY'
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()['features']
    wind_data = pd.DataFrame([obs['properties'] for obs in data])

    # Convert 'observed' column to datetime
    wind_data['observed'] = pd.to_datetime(wind_data['observed'])

    # Set 'observed' as the index
    wind_data.set_index('observed', inplace=True)

    # Select only numeric columns for resampling
    numeric_columns = wind_data.select_dtypes(include=['number']).columns
    wind_data_numeric = wind_data[numeric_columns]

    # Resample to hourly intervals by averaging
    wind_data_resampled = wind_data_numeric.resample('h').mean().reset_index()

    return wind_data_resampled

# Feature Engineering
def preprocess_data(df, wind_data):
    df['hour'] = df['HourDK'].dt.hour
    df['day'] = df['HourDK'].dt.day
    df['month'] = df['HourDK'].dt.month
    df['weekday'] = df['HourDK'].dt.weekday
    df['PriceEUR'] = df['SpotPriceDKK'] / 7.44  # Conversion to EUR

    # Ensure both datetime columns are in the same timezone
    df['HourDK'] = pd.to_datetime(df['HourDK']).dt.tz_localize(None)
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)

    # Merge wind data with electricity prices
    wind_data.rename(columns={'observed': 'HourDK'}, inplace=True)
    df = pd.merge_asof(df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
    df.dropna(inplace=True)
    return df

# Train-Test Split
def split_data(df):
    features = ['hour', 'day', 'month', 'weekday', 'value']  # 'value' is wind speed
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
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'weekday': [d.weekday() for d in future_dates]
    })

    # Fetch real wind speed measurements
    start_date = asof - timedelta(hours=1)
    end_date = start_date + timedelta(days=7)
    wind_data = fetch_wind_data(start_date, end_date)

    future_df['HourDK'] = future_dates

    # Ensure both datetime columns are in the same timezone
    future_df['HourDK'] = pd.to_datetime(future_df['HourDK']).dt.tz_localize(None)
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    
    # Merge wind data with future dates
    wind_data.rename(columns={'observed': 'HourDK'}, inplace=True)
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
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
    df = fetch_data(datetime(2020, 1, 1), asof)
    wind_data = fetch_wind_data(datetime(2020, 1, 1), asof)
    df = preprocess_data(df, wind_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predict_future(model, asof)
