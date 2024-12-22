import requests
import pandas as pd
import duckdb
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


# Print the first 5 rows of the data in the parquet file
con = duckdb.connect('spot_prices.db')
result = con.execute('SELECT * FROM spot_prices LIMIT 5')
print(result.fetchdf())

# Count the number of rows in the parquet file
result = con.execute('SELECT COUNT(*) FROM spot_prices')
print(result.fetchdf())

con.close()