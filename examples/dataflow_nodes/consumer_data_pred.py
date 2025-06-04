# Function for predicting consumer data using PyCaret
import os
import sys
import pandas as pd
from pycaret.regression import load_model, predict_model, compare_models, setup, save_model

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


def predict_consumer_data(dataframe={}, parameters = {"hours": 24, "model_name":"consumer_model", "factor": 1}):
    """
    Predict consumer data using a trained PyCaret model.

    Parameters
    ----------
    dataframe : pd.DataFrame, optional
        Dataframe containing the input data for prediction. If None, an empty dataframe will be created with the necessary features.
    hours : int, optional
        Number of hours to predict. Default is 24.
    model_name : str, optional
        Name of the model to load. Default is "consumer_model".

    Returns
    -------
    pd.DataFrame
        Dataframe containing the predictions.
    """

    # Create a dataframe containing datetime index, hour, dayofweek, and month features for the next n hours
    df = pd.DataFrame({
        'datetime': pd.date_range(start=pd.Timestamp.now().normalize(), periods=parameters["hours"], freq='H')
    })
    df.set_index('datetime', inplace=True)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    # Load the trained model
    try:
        model = load_model(parameters["model_name"])
    except FileNotFoundError:
        model = None

    if model is None:
        train_consumer_model(dataframe=None, model_name=parameters["model_name"])
        model = load_model(parameters["model_name"])

    # Make predictions
    predictions = predict_model(model, data=df)

    predictions = predictions[['prediction_label']].rename(
        columns={'prediction_label': 'consumption_kWh'})
    
    predictions['consumption_kWh'] *= parameters["factor"]
    dataframe["gen_consumption"] = predictions

    return dataframe


def train_consumer_model(dataframe=None, model_name="consumer_model"):
    """
    Train a consumer data prediction model using PyCaret.

    Parameters
    ----------
    dataframe : pd.DataFrame, optional
        Dataframe containing the training data. If None, it will fetch data from the dataflow.
    model_name : str, optional
        Name of the model to save. Default is "consumer_model".

    Returns
    -------
    None
    """
    df = dataframe

    if df is None:
        # Fetch training data if not provided
        df = fetch_training_data()

    # Add time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    # Initialize PyCaret setup with gpu support
    setup(data=df, target='active_energy_kWh', session_id=123, use_gpu=False,n_jobs=2,verbose=True)

    # Find best model with turbo mode enabled
    best_model = compare_models(turbo=True)

    # Save the trained model
    save_model(best_model, model_name)


def fetch_training_data():
    """
    Fetch training data for the consumer model.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the training data.
    """
    df = pd.read_csv(
        'data/household_power_consumption.txt',
        sep=';',
        na_values=['?'],
        low_memory=False,)

    df = df.dropna()  # Drop rows with NaN values
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('datetime', inplace=True)

    # Convert relevant column to numeric
    df['global_active_power'] = pd.to_numeric(
        df['Global_active_power'], errors='coerce')

    # Resample to hourly, summing the kW-minutes, then divide by 60 to get kWh
    df_hourly = df['global_active_power'].resample('h').sum() / 60
    df = df_hourly.to_frame(name='active_energy_kWh')

    return df


if __name__ == "__main__":
    # df = fetch_training_data()
    # train_consumer_model(df)
    predictions = predict_consumer_data()

    # Plot the predictions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(predictions.index,
             predictions['consumption_kWh'], label='Predicted Consumption (kWh)')
    plt.title('Predicted Consumer Data')
    plt.xlabel('Datetime')
    plt.ylabel('Consumption (kWh)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig("predicted_consumer_data.png")
    print(predictions)
