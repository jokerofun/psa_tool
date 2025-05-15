import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

openmeteo = None

def get_client():
    ## Create a singleton instance of the OpenMeteo client
    global openmeteo
    if openmeteo is None:
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
    return openmeteo

def get_wind_data(dataframe = None, parameters: dict = {"latitude": 0, "longitude": 0}):
    """
    Get wind data forecast for 24 hours with 1 hour intervals.
    
    Parameters
    ----------
    dataframe : pd.DataFrame, optional
        Dataframe to store the data. The default is None.
    parameters : dict, optional
        Dictionary with latitude and longitude. The default is {"latitude": 0, "longitude": 0}.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with wind data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": parameters["latitude"],
        "longitude": parameters["longitude"],
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "ms"
    }
    responses = get_client().weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy() 
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    
    return hourly_dataframe

## example usage
if __name__ == "__main__":
    parameters = {"latitude": 57.0488, "longitude": 9.9217}  # Aalborg, Denmark
    wind_data = get_wind_data(parameters=parameters)
    print(wind_data)