import requests
import numpy as np
import cvxpy as cp
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from datetime import datetime

con = duckdb.connect('spot_prices.db')
result = con.execute('SELECT * FROM information_schema.tables WHERE table_name = \'spot_prices\'')
table_exists = len(result.fetchdf()) > 0

if not table_exists:
    # Fetch data from Energinet data portal (since 2020-01-01)
    url = 'https://api.energidataservice.dk/dataset/Elspotprices?offset=0&start=2020-01-01T00:00&end=2024-12-31T00:00&filter=%7B%22PriceArea%22:[%22DK1%22]%7D&sort=HourUTC%20DESC'
    response = requests.get(url)
    result = response.json()
    data = result.get('records', [])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df['PriceArea'] = df['PriceArea'].astype('category')
    df['SpotPriceDKK'] = df['SpotPriceDKK'].astype('float')

    # Store data in parquet file using DuckDB
    con.register('spot_prices_vitual', df)
    con.execute('CREATE TABLE spot_prices AS SELECT * FROM spot_prices_vitual')


# # Print the first 5 rows of the data in the parquet file
# con = duckdb.connect('spot_prices.db')
# result = con.execute('SELECT * FROM spot_prices LIMIT 5')
# print(result.fetchdf())

# # Count the number of rows in the parquet file
# result = con.execute('SELECT COUNT(*) FROM spot_prices')
# print(result.fetchdf())

# Battery specifications (Tesla battery)
battery_capacity = 200  # MWh
charging_power = 100 # MW
discharging_power = 100 # MW
efficiency = 0.9

# Load the future predicted prices from the database
result = con.execute('SELECT * FROM future_prices')
future_df = result.fetchdf()
future_df['HourDK'] = pd.to_datetime(future_df['HourDK'])
future_df.set_index('HourDK', inplace=True)

def battery_arbitrage(prices, battery_capacity, charging_power, discharging_power, efficiency):
    T = len(prices)  # Number of time periods (hours)
    
    # Decision Variables
    charge = cp.Variable(T, nonneg=True)  # Charging power
    discharge = cp.Variable(T, nonneg=True)  # Discharging power
    state_of_charge = cp.Variable(T, nonneg=True)  # Battery state of charge
    binary = cp.Variable(T, boolean=True)  # Binary to control charge/discharge mode

    # Objective: Maximize arbitrage profit
    profit = cp.sum(prices @ discharge) - cp.sum(prices @ charge)
    objective = cp.Maximize(profit)

    # Constraints
    constraints = [
        state_of_charge[0] == 0,  # Start with empty battery
        state_of_charge[-1] == 0  # End with empty battery
    ]

    M = battery_capacity  # Big-M parameter (or max(charging_power, discharging_power))

    for t in range(T):
        if t == 0:
            constraints += [
                state_of_charge[t] == efficiency * charge[t] - (1 / efficiency) * discharge[t]
            ]
        else:
            constraints += [
                state_of_charge[t] == state_of_charge[t-1] + efficiency * charge[t] - (1 / efficiency) * discharge[t]
            ]
        
        # Power and capacity constraints
        constraints += [
            charge[t] <= charging_power,
            discharge[t] <= discharging_power,
            state_of_charge[t] <= battery_capacity,
            
            # Big-M constraints (to prevent charge & discharge at the same time)
            charge[t] <= M * binary[t],                # If binary[t] == 0, charge[t] = 0
            discharge[t] <= M * (1 - binary[t])        # If binary[t] == 1, discharge[t] = 0
        ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    print(f"Optimal Profit: {problem.value:.2f} EUR")
    
    return charge.value, discharge.value, state_of_charge.value

# Run the battery arbitrage strategy
future_prices = future_df['PriceEUR'].values
charge, discharge, soc = battery_arbitrage(future_prices, battery_capacity, charging_power, discharging_power, efficiency)

# Plot the battery arbitrage strategy
plt.figure(figsize=(12, 6))
plt.plot(future_df.index, soc, label='State of Charge (MWh)', color='blue')
plt.bar(future_df.index, charge, width=0.02, label='Charge (MW)', color='green')
plt.bar(future_df.index, -discharge, width=0.02, label='Discharge (MW)', color='red')
plt.legend()
plt.title('Battery Arbitrage Strategy')
plt.xlabel('Date')
plt.ylabel('Energy (MWh)')
plt.show()
plt.savefig('battery_arbitrage.png')
con.close()