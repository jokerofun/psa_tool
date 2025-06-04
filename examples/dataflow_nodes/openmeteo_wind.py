import openmeteo_requests

import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataflow_manager.clients import openmeteo_client

def get_wind_data(dataframe = {}, parameters: dict = {"latitude": 0, "longitude": 0, "T": 24, "l": 1}):
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
    parameters = {**{"latitude": 0, "longitude": 0, "T": 24, "l" : 1}, **parameters}
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": parameters["latitude"],
        "longitude": parameters["longitude"],
        "hourly": "wind_speed_100m",
        "wind_speed_unit": "ms"
    }
    responses = openmeteo_client.get_openmeteo_client().weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy() 
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    if parameters["l"] == 1:
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_dataframe = pd.DataFrame(data = hourly_data)
    elif parameters["l"] == 4:
        # interpolate the data to 15 minute intervals, with weighed average
        hourly_data_interp = pd.DataFrame()
        hourly_data_interp["date"] = hourly_data["date"]
        
        # Create a temporary hourly dataframe
        temp_hourly_df = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=len(hourly_wind_speed_10m),
                freq="H"
            ),
            "wind_speed_10m": hourly_wind_speed_10m
        })
        
        # Resample to 15-minute intervals with linear interpolation
        resampled = temp_hourly_df.set_index("date").resample('15min').interpolate(method='linear')
        
        # Add to hourly_data
        hourly_dataframe = resampled.reset_index()

    dataframe["get_wind_data"] = hourly_dataframe
    
    return dataframe

## example usage
if __name__ == "__main__":
    parameters = {"latitude": 57.0488, "longitude": 9.9217, "l": 4}  # Aalborg, Denmark
    wind_data = get_wind_data(parameters=parameters)
    print(wind_data)