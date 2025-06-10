import json
import pandas as pd
import requests


def fetch_spot_prices(dfs, parameters):
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        'start': parameters['start_date'].strftime('%Y-%m-%dT%H:%M'),
        'end': parameters['end_date'].strftime('%Y-%m-%dT%H:%M'),
        'filter': json.dumps({"PriceArea": "DK1"})
    }
    response = requests.get(url, params=params)
    data = response.json()['records']
    df = pd.DataFrame(data)
    df['HourDK'] = pd.to_datetime(df['HourDK'])

    df["SpotPriceDKK"] = df["SpotPriceDKK"] / 1000  # Convert to DKK/kWh
    df["SpotPriceEUR"] = df["SpotPriceEUR"] / 1000  # Convert to EUR/kWh

    dfs["spot_prices"] = df

    return dfs