import requests
import pandas as pd
import duckdb

from lib_descriptive import plot_battery_arbitrage, plot_battery_arbitrage_multiple
from lib_prescriptive import battery_arbitrage, battery_arbitrage_multiple

import matplotlib.pyplot as plt

con = duckdb.connect('database.db')
result = con.execute('SELECT * FROM information_schema.tables WHERE table_name = \'future_prices\'')
table_exists = len(result.fetchdf()) > 0

if not table_exists:
    # Fetch data from Energinet data portal (since 2020-01-01)
    url = 'https://api.energidataservice.dk/dataset/Elspotprices?offset=0&start=2020-01-01T00:00&end=2024-12-08T00:00&filter=%7B%22PriceArea%22:[%22DK1%22]%7D&sort=HourUTC%20DESC'
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

# Number of days to query
start_date = '2024-06-05'
number_of_days = 1



# Load the future predicted prices from the database
result = con.execute('SELECT * FROM future_prices')
future_df = result.fetchdf()

# Cut from start_date and add number of days
future_df = future_df[future_df['HourDK'] >= start_date]
future_df = future_df[future_df['HourDK'] < pd.to_datetime(start_date) + pd.DateOffset(days=number_of_days)]

future_df['HourDK'] = pd.to_datetime(future_df['HourDK'])
future_df.set_index('HourDK', inplace=True)
print(future_df.head())
print(future_df.tail())
# Run the battery arbitrage strategy
future_prices = future_df['SpotPriceEUR'].values


optimal_profit, optimal_schedule, soc_schedule = battery_arbitrage_multiple(future_prices, num_days=number_of_days, num_batteries=1)

# Plot the battery arbitrage strategy
# plot_battery_arbitrage_multiple(future_prices, soc_schedule, optimal_schedule, num_batteries=2)



con.close()