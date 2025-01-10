import matplotlib.pyplot as plt

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