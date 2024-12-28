import pandas as pd
import numpy as np
from duckdb import connect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from hyperopt import hp, fmin, tpe, Trials
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import tensorflow as tf
import keras
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense

# Import mse from keras metrics
# from tensorflow.keras.metrics import MeanSquaredError

API_KEY = 'YOUR_DMI_API_KEY'
STATION_ID = '06030'  # Aalborg weather station
LAT, LON = 57.048, 9.9187 # Coordinates for Aalborg

# DMI Data Retrieval Function
def fetch_data(url, start_date, end_date, parameter, historical=False):
    if historical:
        # Read data from csv file if it exists
        try:
            df = pd.read_csv(f'{parameter}_test.csv')
            df['observed'] = pd.to_datetime(df['observed'])
            return df
        except FileNotFoundError:
            pass
    else:
        params = {
            'datetime': f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            'parameterId': parameter,
            'stationId': STATION_ID,
            'limit': 300000
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

# Fetch Forecast Data from Open-Meteo
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

# Spot Price Data Retrieval Function
def fetch_spot_prices(start_date, end_date):
    # Read data from csv file if it exists
    try:
        df = pd.read_csv('spot_prices_test.csv')
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

# Feature Engineering
def preprocess_data(df, wind_data, temp_data):
    df['hour'] = df['HourDK'].dt.hour
    df['weekday'] = df['HourDK'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    df['PriceEUR'] = df['SpotPriceEUR']
    df['PriceEUR_lag1'] = df['PriceEUR'].shift(1)
    df['PriceEUR_lag24'] = df['PriceEUR'].shift(24)
    df['PriceEUR_rolling_mean'] = df['PriceEUR'].rolling(24).mean()

    df = df.drop(columns=['hour', 'weekday', 'HourUTC', 'SpotPriceDKK', 'PriceArea', 'SpotPriceEUR'])

    # Ensure both datetime columns are in the same timezone
    df['HourDK'] = pd.to_datetime(df['HourDK']).dt.tz_localize(None)
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    temp_data['observed'] = pd.to_datetime(temp_data['observed']).dt.tz_localize(None)

    wind_data.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    temp_data.rename(columns={'observed': 'HourDK', 'value': 'temperature'}, inplace=True)

    # Merge wind data with electricity prices
    df = pd.merge_asof(df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
    df = pd.merge_asof(df.sort_values('HourDK'), temp_data.sort_values('HourDK'), on='HourDK')
    df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
    df.dropna(inplace=True)

    # # Remove outliers using IQR
    # Q1 = df['PriceEUR'].quantile(0.25)
    # Q3 = df['PriceEUR'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df = df[(df['PriceEUR'] >= lower_bound) & (df['PriceEUR'] <= upper_bound)]

    return df

# Train-Test Split
def split_data(df):
    features = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'PriceEUR_lag1', 'PriceEUR_lag24', 'PriceEUR_rolling_mean', 'wind_speed', 'temperature', 'wind_temp_interaction']
    target = 'PriceEUR'
    X = df[features]
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Reshape Data for LSTM
def reshape_data(X_train, X_val, X_test):
    X_train = X_train.values.reshape((X_train.shape[0], 24, X_train.shape[1]))
    X_val = X_val.values.reshape((X_val.shape[0], 24, X_val.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 24, X_test.shape[1]))
    return X_train, X_val, X_test

timesteps = 24  # Or 168 for a week

def reshape_data2(X, y, timesteps):
    samples = X.shape[0] - timesteps
    reshaped_X = np.array([X[i:i + timesteps] for i in range(samples)])
    reshaped_y = y[timesteps:]  # Align y to match reshaped X
    return reshaped_X, reshaped_y

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM Model
def train_lstm_model(X_train, y_train, X_val, y_val):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2, shuffle=False)
    model.save('lstm_model.keras')  # Save the model
    return model, history

# Load the model
def load_model():
    return keras.models.load_model('lstm_model.keras', custom_objects={'mse': 'mse'})

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
        'weekday': [d.weekday() for d in future_dates],
        'hour_sin': np.sin(2 * np.pi * np.array([d.hour for d in future_dates])/24),
        'hour_cos': np.cos(2 * np.pi * np.array([d.hour for d in future_dates])/24),
        'weekday_sin': np.sin(2 * np.pi * np.array([d.weekday() for d in future_dates])/7),
        'weekday_cos': np.cos(2 * np.pi * np.array([d.weekday() for d in future_dates])/7),
        'PriceEUR_lag1': np.nan,
        'PriceEUR_lag24': np.nan,
        'PriceEUR_rolling_mean': np.nan,
    })

    future_df = future_df.drop(columns=['hour', 'weekday'])

    # Fetch forecast data for the next 7 days
    forecast_df = fetch_forecast_data(LAT, LON)

    # Merge forecast data with future dates
    forecast_df['HourDK'] = forecast_df['HourDK'].dt.tz_localize(None)
    future_df['HourDK'] = future_dates
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), forecast_df.sort_values('HourDK'), on='HourDK')
    future_df['wind_temp_interaction'] = future_df['wind_speed'] * future_df['temperature']

    # Drop the 'HourDK' column before making predictions
    future_df = future_df.drop(columns=['HourDK'])

    print(future_df.shape)

    # Reshape future_df for LSTM
    # future_df = future_df.values.reshape((future_df.shape[0], 1, future_df.shape[1]))
    future_df = future_df.values.reshape((future_df.shape[0], 1, future_df.shape[1]))

    print(future_df.shape)

    # Make predictions
    future_predictions = model.predict(future_df)

    # Convert predictions to DataFrame and reattach HourDK
    future_predictions_df = pd.DataFrame(future_predictions, columns=['PriceEUR'])
    future_predictions_df['HourDK'] = future_dates

    # Plotting Future Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions_df['HourDK'], future_predictions_df['PriceEUR'], label='Future Predictions', color='green')
    plt.title('Predicted Spot Prices for the Next 7 Days (DK1)')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.legend()
    plt.show()
    return future_df, future_predictions

# Create histograms for train and test data
def plot_histograms(y_train, y_test):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=30, color='blue', alpha=0.7)
    plt.title('Training Data Distribution')
    plt.xlabel('PriceEUR')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(y_test, bins=30, color='green', alpha=0.7)
    plt.title('Test Data Distribution')
    plt.xlabel('PriceEUR')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Normalize Data
def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y

# Main Execution
if __name__ == "__main__":
    asof = datetime.combine(datetime.now(), time.min)
    start_date = datetime(2020, 1, 1)

    df = fetch_spot_prices(start_date, asof)
    wind_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'wind_speed', historical=True)
    temp_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'temp_dry', historical=True)

    df = preprocess_data(df, wind_data, temp_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Normalize data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print(df.head())

    # Plot histograms
    # plot_histograms(y_train, y_test)

    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    # Reshape data for LSTM
    # X_train, X_val, X_test = reshape_data(X_train, X_val, X_test)
    X_train, y_train = reshape_data2(X_train, y_train, timesteps)
    X_val, y_val = reshape_data2(X_val, y_val, timesteps)
    X_test, y_test = reshape_data2(X_test, y_test, timesteps)

    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    # # Train LSTM model
    # model, history = train_lstm_model(X_train, y_train, X_val, y_val)

    # # model.fit(X_train, y_train)
    # evaluate_model(model, X_test, y_test)

    # Load the model for future predictions
    model = load_model()
    future_df, predictions = predict_future(model, asof)

    # # Inverse transform the predictions to get the original scale
    future_df['PriceEUR'] = scaler_y.inverse_transform(future_df[['PriceEUR']])
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
    # print(predictions)
    # print(future_df)