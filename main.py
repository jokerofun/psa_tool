import requests
import pandas as pd
import duckdb
from datetime import datetime

# Fetch data from Energinet data portal
url = 'https://api.energidataservice.dk/dataset/Elspotprices?limit=8760'
response = requests.get(url)
result = response.json()
data = result.get('records', [])

# Convert data to pandas DataFrame
df = pd.DataFrame(data)
df['HourDK'] = pd.to_datetime(df['HourDK'])
df['PriceArea'] = df['PriceArea'].astype('category')
df['SpotPriceDKK'] = df['SpotPriceDKK'].astype('float')

# Store data in parquet file using DuckDB
con = duckdb.connect('spot_prices.db')
con.register('spot_prices', df)
con.execute('CREATE TABLE spot_prices AS SELECT * FROM spot_prices')
con.close()

# Print the first 5 rows of the data in the parquet file (only DK1)
con = duckdb.connect('spot_prices.db')
result = con.execute('SELECT * FROM spot_prices LIMIT 5')
print(result.fetchdf())
