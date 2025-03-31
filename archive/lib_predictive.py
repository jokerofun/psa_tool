import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import time, timedelta
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout

import datetime

from lib_data import fetch_data
from lib_descriptive import plot_future_predictions, plot_predictions

# Function to train an LSTM model
def train_lstm_model(X_train, y_train, X_val, y_val):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val), verbose=2, shuffle=False)
    model.save('lstm_model.keras')  # Save the model
    return model, history

# Function to build an LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(50, activation='relu', dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='huber')
    return model

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.2f} EUR")
    return predictions

# Function to preprocess data
def preprocess_data(df, wind_array, temp_data, gas_data):
    df['hour'] = df['HourDK'].dt.hour
    df['epoch'] = df['HourDK'].astype(np.int64) 
    df['weekday'] = df['HourDK'].dt.weekday
    df['day_of_year'] = df['HourDK'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    df['PriceEUR'] = df['SpotPriceEUR']

    df = df.drop(columns=[ 'HourUTC', 'SpotPriceDKK', 'PriceArea', 'SpotPriceEUR'])
    #df = df.drop(columns=['hour', 'weekday', 'HourUTC', 'SpotPriceDKK', 'PriceArea', 'SpotPriceEUR'])

    # Ensure both datetime columns are in the same timezone
    df['HourDK'] = pd.to_datetime(df['HourDK']).dt.tz_localize(None)
    # wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    for wind in wind_array:
        wind['observed'] = pd.to_datetime(wind['observed']).dt.tz_localize(None)
    temp_data['observed'] = pd.to_datetime(temp_data['observed']).dt.tz_localize(None)

    # wind_data.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    for wind in wind_array:
        # rename columns to avoid duplicates
        wind.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    temp_data.rename(columns={'observed': 'HourDK', 'value': 'temperature'}, inplace=True)

    # Merge wind data with electricity prices
    for wind in wind_array:
        df = pd.merge_asof(df.sort_values('HourDK'), wind.sort_values('HourDK'), on='HourDK')
    df = pd.merge_asof(df.sort_values('HourDK'), temp_data.sort_values('HourDK'), on='HourDK')
    df = pd.merge_asof(df.sort_values('HourDK'), gas_data.sort_values('HourDK'), on='HourDK')
    df.dropna(inplace=True)
    
    # Remove outliers with prices above 300 Eur, replace with 300 Eur
    #df = df[df['PriceEUR'] < 300]
    df.loc[df['PriceEUR'] > 300, 'PriceEUR'] = 300

    # # Remove outliers using IQR
    # Q1 = df['PriceEUR'].quantile(0.25)
    # Q3 = df['PriceEUR'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df = df[(df['PriceEUR'] >= lower_bound) & (df['PriceEUR'] <= upper_bound)]

    return df

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

# Function to split data into training, validation, and test sets
def split_data(df, features, target):
    # features = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'wind_speed', 'temperature']
    # target = 'PriceEUR'
    X = df[features]
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42069, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42069, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to reshape data for LSTM
def reshape_data_LSTM(X_train, X_val, X_test):
    # if x_train has attribute value 
    if hasattr(X_train, 'values'):
        X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    else:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train, X_val, X_test

# Function to predict future prices
def predict_future(model, asof):
    future_dates = [asof + timedelta(hours=i) for i in range(24*7)]
    future_df = pd.DataFrame({
        'day_of_year': [d.timetuple().tm_yday for d in future_dates],
        'weekday': [d.weekday() for d in future_dates],
        'hour': [d.hour for d in future_dates],
        'hour_sin': np.sin(2 * np.pi * np.array([d.hour for d in future_dates])/24),
        'hour_cos': np.cos(2 * np.pi * np.array([d.hour for d in future_dates])/24),
        'weekday_sin': np.sin(2 * np.pi * np.array([d.weekday() for d in future_dates])/7),
        'weekday_cos': np.cos(2 * np.pi * np.array([d.weekday() for d in future_dates])/7),
        'epoch' : [int(d.timestamp()) for d in future_dates],
    })

    future_df = future_df.drop(columns=['hour', 'weekday'])


    # Fetch forecast data for the next 7 days
    # forecast_df = fetch_forecast_data(LAT, LON)
    wind_data = pd.read_csv('data/wind_speed_1201_1208.csv')
    temp_data = pd.read_csv('data/temp_dry_1201_1208.csv')

    # Ensure both datetime columns are in the same timezone
    wind_data['observed'] = pd.to_datetime(wind_data['observed']).dt.tz_localize(None)
    temp_data['observed'] = pd.to_datetime(temp_data['observed']).dt.tz_localize(None)

    wind_data.rename(columns={'observed': 'HourDK', 'value': 'wind_speed'}, inplace=True)
    temp_data.rename(columns={'observed': 'HourDK', 'value': 'temperature'}, inplace=True)

    # Merge forecast data with future dates
    future_df['HourDK'] = future_dates
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), wind_data.sort_values('HourDK'), on='HourDK')
    future_df = pd.merge_asof(future_df.sort_values('HourDK'), temp_data.sort_values('HourDK'), on='HourDK')
    # future_df = pd.merge_asof(future_df.sort_values('HourDK'), forecast_df.sort_values('HourDK'), on='HourDK')

    # Drop the 'HourDK' column before making predictions
    future_df = future_df.drop(columns=['HourDK'])

    # Reshape future_df for LSTM
    future_data = future_df.values.reshape((future_df.shape[0], 1, future_df.shape[1]))

    # Make predictions
    future_predictions = model.predict(future_data)

    # Convert predictions to DataFrame and reattach HourDK
    future_predictions_df = pd.DataFrame(future_predictions, columns=['PriceEUR'])
    future_predictions_df['HourDK'] = future_dates

    # Plotting Future Predictions
    plot_future_predictions(future_predictions_df)

    return future_predictions_df

def load_model(model_path):
    return keras.models.load_model(model_path, custom_objects={'mse': 'mse'})


# example usage
if __name__ == '__main__':
    asof = datetime.combine(datetime.now(), time.min)
    start_date = datetime(2020, 1, 1)

    df = fetch_spot_prices(start_date, asof)
    wind_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'wind_speed', historical=True)
    temp_data = fetch_data("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", start_date, asof, 'temp_dry', historical=True)

    df = preprocess_data(df, wind_data, temp_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Normalize data
    #X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

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
    model, history = train_lstm_model(X_train, y_train, X_val, y_val)

    # # model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Load the model for future predictions
    # model = load_model()
    future_df, predictions = predict_future(model, asof)