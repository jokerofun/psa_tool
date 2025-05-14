import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def get_irradiation_data(dataframe = None, parameters: dict = {"latitude": 0, "longitude": 0}):
    """
    Get irradiation data forecast for 24 hours with 1 hour intervals.
    
    Parameters
    ----------
    dataframe : pd.DataFrame, optional
        Dataframe to store the data. The default is None.
    parameters : dict, optional
        Dictionary with latitude and longitude. The default is {"latitude": 0, "longitude": 0}.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with irradiation data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": parameters["latitude"],
        "longitude": parameters["longitude"],
        "hourly": "direct_radiation",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_irradiation = hourly.Variables(0).ValuesAsNumpy() 
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["direct_irradiation"] = hourly_irradiation

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    
    return hourly_dataframe

## example usage
if __name__ == "__main__":
    parameters = {"latitude": 57.0488, "longitude": 9.9217}  # Aalborg, Denmark
    irradiation_data = get_irradiation_data(parameters=parameters)
    print(irradiation_data)