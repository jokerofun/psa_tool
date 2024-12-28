import requests
import pandas as pd
import duckdb

from lib_descriptive import plot_battery_arbitrage
from lib_prescriptive import battery_arbitrage

con = duckdb.connect('database.db')
result = con.execute('SELECT * FROM information_schema.tables WHERE table_name = \'future_prices\'')
table_exists = len(result.fetchdf()) > 0

if not table_exists:
    # Fetch data from Energinet data portal (since 2020-01-01)
    url = 'https://api.energidataservice.dk/dataset/Elspotprices?offset=0&start=2024-12-01T00:00&end=2024-12-08T00:00&filter=%7B%22PriceArea%22:[%22DK1%22]%7D&sort=HourUTC%20DESC'
    response = requests.get(url)
    result = response.json()
    data = result.get('records', [])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df['PriceArea'] = df['PriceArea'].astype('category')
    df['SpotPriceDKK'] = df['SpotPriceDKK'].astype('float')

    # Store data in parquet file using DuckDB
    con.register('future_prices_vitual', df)
    con.execute('CREATE TABLE future_prices AS SELECT * FROM future_prices_vitual')

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

# Run the battery arbitrage strategy
future_prices = future_df['SpotPriceEUR'].values
charge, discharge, soc = battery_arbitrage(future_prices, battery_capacity, charging_power, discharging_power, efficiency)

# Plot the battery arbitrage strategy
plot_battery_arbitrage(future_df, soc, charge, discharge)

con.close()