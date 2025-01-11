import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to plot histograms of the training and test data
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

# Function plotting Actual vs Prediction
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.legend()
    plt.title('Actual vs Predicted Spot Prices (DK1)')
    plt.xlabel('Test Samples')
    plt.ylabel('Price (EUR)')
    plt.show()

# Function to plot future predictions
def plot_future_predictions(future_predictions_df):
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions_df['HourDK'], future_predictions_df['PriceEUR'], label='Predicted Prices', color='green')
    plt.title('Predicted Spot Prices for the Next 7 Days (DK1)')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.legend()
    plt.show()

# Function for plotting battery arbitrage strategy
def plot_battery_arbitrage(future_df, soc, charge, discharge):
    plt.figure(figsize=(12, 6))
    plt.plot(future_df.index, soc, label='State of Charge (MWh)', color='blue')
    plt.bar(future_df.index, charge, width=0.02, label='Charge (MW)', color='green')
    plt.bar(future_df.index, -discharge, width=0.02, label='Discharge (MW)', color='red')
    plt.legend()
    plt.title('Battery Arbitrage Strategy')
    plt.xlabel('Date')
    plt.ylabel('Energy (MWh)')
    plt.show()


def plot_battery_arbitrage_multiple(prices, soc, schedule, num_batteries):
    """
    Plots the battery arbitrage strategy for the selected batteries.

    Parameters:
        prices (list): Electricity prices over the time periods.
        soc (ndarray): State of charge for each battery over the time periods (T x num_batteries).
        schedule (ndarray): Charging/discharging schedule for each battery (T x num_batteries).
        num_batteries (int): Number of batteries used in the optimization.
    """
    T = len(prices)
    time_index = np.arange(T)

    # Create a DataFrame for convenience
    data = {
        "Prices (EUR/MWh)": prices,
    }
    for i in range(num_batteries):
        data[f"Battery {i+1} SOC (MWh)"] = soc[:, i]
        data[f"Battery {i+1} Charge (MW)"] = np.maximum(0, schedule[:, i])  # Positive values are charging
        data[f"Battery {i+1} Discharge (MW)"] = -np.minimum(0, schedule[:, i])  # Negative values are discharging

    df = pd.DataFrame(data, index=time_index)

    # Plot electricity prices
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df["Prices (EUR/MWh)"], label="Electricity Prices", color="black", linestyle="--", linewidth=1.5)

    # Plot SOC, charging, and discharging for each battery
    for i in range(num_batteries):
        plt.plot(df.index, df[f"Battery {i+1} SOC (MWh)"], label=f"Battery {i+1} SOC", linewidth=2)
        plt.bar(df.index, df[f"Battery {i+1} Charge (MW)"], width=0.4, label=f"Battery {i+1} Charge", color="green", alpha=0.6)
        plt.bar(df.index, -df[f"Battery {i+1} Discharge (MW)"], width=0.4, label=f"Battery {i+1} Discharge", color="red", alpha=0.6)

    # Add titles and labels
    plt.title("Battery Arbitrage Strategy", fontsize=16)
    plt.xlabel("Time Periods", fontsize=14)
    plt.ylabel("Energy (MWh) / Power (MW)", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


# # Function to plot the feature importances
# def plot_feature_importances(model, features):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     plt.figure(figsize=(10, 6))
#     plt.bar(range(features.shape[1]), importances[indices], align='center')
#     plt.xticks(range(features.shape[1]), features.columns[indices], rotation=90)
#     plt.title('Feature Importances')
#     plt.xlabel('Features')
#     plt.ylabel('Importance')
#     plt.show()