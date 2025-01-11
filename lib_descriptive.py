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


# def plot_battery_arbitrage_multiple(prices, soc, schedule, num_batteries):
#     """
#     Plots the battery arbitrage strategy for the selected batteries.

#     Parameters:
#         prices (list): Electricity prices over the time periods.
#         soc (ndarray): State of charge for each battery over the time periods (T x num_batteries).
#         schedule (ndarray): Charging/discharging schedule for each battery (T x num_batteries).
#         num_batteries (int): Number of batteries used in the optimization.
#     """
#     T = len(prices)
#     time_index = np.arange(T)

#     # Create a DataFrame for convenience
#     data = {
#         "Prices (EUR/MWh)": prices,
#     }
#     for i in range(num_batteries):
#         data[f"Battery {i+1} SOC (MWh)"] = soc[:, i]
#         data[f"Battery {i+1} Charge (MW)"] = np.maximum(0, schedule[:, i])  # Positive values are charging
#         data[f"Battery {i+1} Discharge (MW)"] = -np.minimum(0, schedule[:, i])  # Negative values are discharging

#     df = pd.DataFrame(data, index=time_index)

#     # Plot electricity prices
#     plt.figure(figsize=(14, 8))
#     plt.plot(df.index, df["Prices (EUR/MWh)"], label="Electricity Prices", color="black", linestyle="--", linewidth=1.5)

#     # Plot SOC, charging, and discharging for each battery
#     for i in range(num_batteries):
#         plt.plot(df.index, df[f"Battery {i+1} SOC (MWh)"], label=f"Battery {i+1} SOC", linewidth=2)
#         plt.bar(df.index, df[f"Battery {i+1} Charge (MW)"], width=0.4, label=f"Battery {i+1} Charge", color="green", alpha=0.6)
#         plt.bar(df.index, -df[f"Battery {i+1} Discharge (MW)"], width=0.4, label=f"Battery {i+1} Discharge", color="red", alpha=0.6)

#     # Add titles and labels
#     plt.title("Battery Arbitrage Strategy", fontsize=16)
#     plt.xlabel("Time Periods", fontsize=14)
#     plt.ylabel("Energy (MWh) / Power (MW)", fontsize=14)
#     plt.legend(loc="upper left", fontsize=10)
#     plt.grid(True, linestyle="--", alpha=0.7)

#     # Show the plot
#     plt.show()

import os

def plot_battery_arbitrage_multiple(prices, soc, schedule, num_batteries, output_dir="figures"):
    """
    Plots the battery arbitrage strategy for the selected batteries in a 2x2 grid
    and saves each subplot as a PNG file, including electricity prices.

    Parameters:
        prices (list): Electricity prices over the time periods.
        soc (ndarray): State of charge for each battery over the time periods (T x num_batteries).
        schedule (ndarray): Charging/discharging schedule for each battery (T x num_batteries).
        num_batteries (int): Number of batteries used in the optimization.
        output_dir (str): Directory to save the plots as PNG files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

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

    # Prepare the 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Loop through each battery and create a subplot
    for i in range(num_batteries):
        ax = axes[i // 2, i % 2]
        ax.plot(df.index, df[f"Battery {i+1} SOC (MWh)"], label=f"Battery {i+1} SOC", linewidth=2)
        ax.bar(df.index, df[f"Battery {i+1} Charge (MW)"], width=0.4, label=f"Battery {i+1} Charge", color="green", alpha=0.6)
        ax.bar(df.index, -df[f"Battery {i+1} Discharge (MW)"], width=0.4, label=f"Battery {i+1} Discharge", color="red", alpha=0.6)
        ax.plot(df.index, df["Prices (EUR/MWh)"], label="Electricity Prices", color="black", linestyle="--", linewidth=1.5)

        ax.set_title(f"Battery {i+1} Strategy", fontsize=14)
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("Energy (MWh) / Power (MW)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Save only the current subplot as a PNG file
        individual_fig = plt.figure(figsize=(8, 5))
        individual_ax = individual_fig.add_subplot(111)
        individual_ax.plot(df.index, df[f"Battery {i+1} SOC (MWh)"], label=f"Battery {i+1} SOC", linewidth=2)
        individual_ax.bar(df.index, df[f"Battery {i+1} Charge (MW)"], width=0.4, label=f"Battery {i+1} Charge", color="green", alpha=0.6)
        individual_ax.bar(df.index, -df[f"Battery {i+1} Discharge (MW)"], width=0.4, label=f"Battery {i+1} Discharge", color="red", alpha=0.6)
        individual_ax.plot(df.index, df["Prices (EUR/MWh)"], label="Electricity Prices", color="black", linestyle="--", linewidth=1.5)

        individual_ax.set_title(f"Battery {i+1} Strategy", fontsize=14)
        individual_ax.set_xlabel("Time Periods")
        individual_ax.set_ylabel("Energy (MWh) / Power (MW)")
        individual_ax.legend(loc="upper left", fontsize=9)
        individual_ax.grid(True, linestyle="--", alpha=0.7)

        subplot_filename = os.path.join(output_dir, f"battery_{i+1}_strategy.png")
        individual_fig.savefig(subplot_filename, bbox_inches="tight")
        plt.close(individual_fig)  # Close the individual plot to save memory
        print(f"Saved: {subplot_filename}")

    # Create the summary plot in the bottom-right corner
    ax_summary = axes[1, 1]
    ax_summary.plot(df.index, df["Prices (EUR/MWh)"], label="Electricity Prices", color="black", linestyle="--", linewidth=1.5)

    # Aggregate SOC, charging, and discharging for all batteries
    total_soc = soc.sum(axis=1)
    total_charge = np.maximum(0, schedule).sum(axis=1)
    total_discharge = -np.minimum(0, schedule).sum(axis=1)

    ax_summary.plot(df.index, total_soc, label="Total SOC (MWh)", linewidth=2)
    ax_summary.bar(df.index, total_charge, width=0.4, label="Total Charge (MW)", color="green", alpha=0.6)
    ax_summary.bar(df.index, -total_discharge, width=0.4, label="Total Discharge (MW)", color="red", alpha=0.6)

    ax_summary.set_title("Summary Plot", fontsize=14)
    ax_summary.set_xlabel("Time Period (in hours)")
    ax_summary.set_ylabel("Energy (MWh) / Power (MW)")
    ax_summary.legend(loc="upper left", fontsize=9)
    ax_summary.grid(True, linestyle="--", alpha=0.7)

    # Hide any unused subplots if num_batteries < 4
    for j in range(num_batteries, 4):
        fig.delaxes(axes[j // 2, j % 2])

    # Adjust layout and show the 2x2 grid
    plt.tight_layout()
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

# Function to plot execution time of different phases of analytics
def plot_exec_time(DA_exec_time, PDA_create_model_exec_time, PDA_predict_exec_time, PSA_exec_time):
    categories = ['Data Processing', 'Predictions', 'Prescriptions']

    PDA_exec_time = PDA_create_model_exec_time + PDA_predict_exec_time
    times = [DA_exec_time, PDA_exec_time, PSA_exec_time]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, times, color=['blue', 'orange', 'green'], edgecolor='black')
    ax.bar('Predictions', PDA_create_model_exec_time, color='yellow', edgecolor='black', label='Create Model')
    ax.bar('Predictions', PDA_predict_exec_time, bottom=PDA_create_model_exec_time, color='red', edgecolor='black', label='Forecast')
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_PSA_exec_time(data):
    """
    Create a bar chart with execution times.
    
    Args:
        data (list of dict): A list of dictionaries, where each dictionary has:
            - "days": numeric value representing the number of days.
            - "batteries": numeric value representing the number of batteries.
            - "exec_time": numeric value representing the execution time in seconds.
    """
    # Generate labels and heights from the input data
    labels = [f"{item['days']} day(s), {item['batteries']} batteries" for item in data]
    heights = [item['exec_time'] for item in data]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, heights, color='skyblue', edgecolor='black')
    
    # Add labels, title, and grid
    plt.xlabel("Number of Days and Batteries", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.title("Battery Arbitrage Execution Times", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the chart
    plt.tight_layout()
    plt.show()
