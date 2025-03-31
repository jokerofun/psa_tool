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
            'datetime': f'{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}',
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

    df = df.drop(columns=['hour', 'weekday'])

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

    # Remove outliers using IQR
    Q1 = df['PriceEUR'].quantile(0.25)
    Q3 = df['PriceEUR'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['PriceEUR'] >= lower_bound) & (df['PriceEUR'] <= upper_bound)]

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

    future_predictions = model.predict(future_df)

    # Reattach HourDK to predictions
    future_df['PriceEUR'] = future_predictions
    future_df['HourDK'] = future_dates

    # Set datetime as index for easier visualization
    future_df.set_index('HourDK', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green')
    plt.title('Predicted Spot Prices for the Next 7 Days (DK1)')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.legend()
    plt.show()
    return future_df, future_predictions

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

# Hyperparameter Space for XGBoost
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'subsample' : hp.uniform('subsample', 0.5, 1),
        'max_delta_step' : hp.quniform('max_delta_step', 0, 10, 1),
        'learning_rate' : hp.uniform('learning_rate', 0.01, 0.3),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform("n_estimators", 100, 1000, 50),
        # 'seed': 0
    }

# Objective Function for Hyperparameter Optimization
def objective(params):
    model = XGBRegressor(
        max_depth=int(params['max_depth']),
        gamma=params['gamma'],
        reg_alpha=int(params['reg_alpha']),
        reg_lambda=params['reg_lambda'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=int(params['min_child_weight']),
        n_estimators=int(params['n_estimators']),
        subsample=params['subsample'],
        max_delta_step=int(params['max_delta_step']),
        learning_rate=params['learning_rate'],
        early_stopping_rounds=10,
        # seed=int(params['seed'])
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    return mae

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
    # X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # Plot histograms
    plot_histograms(y_train, y_test)

    # # Hyperparameter Optimization
    # trials = Trials()

    # best_params = fmin(
    #     fn=objective,
    #     space=space,
    #     algo=tpe.suggest,
    #     max_evals=100,
    #     trials=trials
    # )

    # print("Best Hyperparameters:", best_params)

    # # Train model with best hyperparameters
    # optimized_model = XGBRegressor(
    #     max_depth=int(best_params['max_depth']),
    #     gamma=best_params['gamma'],
    #     reg_alpha=int(best_params['reg_alpha']),
    #     reg_lambda=best_params['reg_lambda'],
    #     colsample_bytree=best_params['colsample_bytree'],
    #     min_child_weight=int(best_params['min_child_weight']),
    #     n_estimators=int(best_params['n_estimators']),
    #     subsample=best_params['subsample'],
    #     max_delta_step=int(best_params['max_delta_step']),
    #     learning_rate=best_params['learning_rate'],
    #     # seed=int(best_params['seed'])
    # )

    # optimized_model = XGBRegressor(
    #     max_depth=18,
    #     gamma=4.631882543711501,
    #     reg_alpha=46,
    #     reg_lambda=0.8233384474408272,
    #     colsample_bytree=0.9468464912765524,
    #     min_child_weight=10,
    #     n_estimators=1000,
    #     subsample=0.5557340216529043,
    #     max_delta_step=4,
    #     learning_rate=0.20017834553594877,
    # )

    optimized_model = XGBRegressor(
        max_depth=12,
        gamma=1.3306308847945991,
        reg_alpha=142,
        reg_lambda=0.3981270988816685,
        colsample_bytree=0.9269009296607986,
        min_child_weight=3,
        n_estimators=150,
        subsample=0.8522180029456405,
        max_delta_step=0.0,
        learning_rate=0.07535437517076035,
    )

    optimized_model.fit(X_train, y_train)
    evaluate_model(optimized_model, X_test, y_test)

    future_df, future_predictions = predict_future(optimized_model, asof)

    # # Inverse transform the predictions to get the original scale
    # future_df['PriceEUR'] = scaler_y.inverse_transform(future_df[['PriceEUR']])
    # future_predictions = scaler_y.inverse_transform(future_predictions.reshape(-1, 1))

    print(future_df)

    # # Save future predictions to DB using DuckDB including the index (DuckDB does not include the index by default)
    # future_df.reset_index(inplace=True)

    # con = connect('spot_prices.db')
    # con.register('future_prices_virtual', future_df)

    # # If table exists already, drop it and insert new data
    # result = con.execute('SELECT * FROM information_schema.tables WHERE table_name = \'spot_prices\'')
    # table_exists = len(result.fetchdf()) > 0

    # if table_exists:
    #     con.execute('DELETE FROM future_prices')
    #     con.execute('INSERT INTO future_prices SELECT * FROM future_prices_virtual')
    # else:
    #     con.execute('CREATE TABLE future_prices AS SELECT * FROM future_prices_virtual')

    # con.close()