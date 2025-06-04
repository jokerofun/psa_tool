import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_consumption_data(dataframe = {}, parameters: dict = {"A0" : 0, "A1": 0, "A2": 0, "phi0": 0, "phi1": 0, "T": 24, "l": 1}):
    """
    Generate consumption data for 24 hours with 1 hour intervals.
    
    Parameters
    ----------
    parameters : dict, optional
        Dictionary with A0, A1, A2, phi0 and phi1. The default is {"A0" : 0, "A1": 0, "A2": 0, "phi0": 0, "phi1": 0, "T": 24, "l": 1}.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with consumption data.
    """
    # merge parameters dict with default values, but only if they are not already set
    parameters = {**{"A0": 0, "A1": 0, "A2": 0, "phi0": 0, "phi1": 0, "T": 24, "l": 1}, **parameters}
    

    df = pd.DataFrame()
    # Generate time series for 24 hours with 1 hour intervals
    if parameters["l"] == 1:
        df["date"] = pd.date_range(start = pd.Timestamp.now(), periods = parameters["T"], freq = "h")
    elif parameters["l"] == 4:
        df["date"] = pd.date_range(start = pd.Timestamp.now(), periods = parameters["T"]*parameters["l"], freq = "15min")
    # Generate consumption data using the parameters
    A0 = parameters["A0"]
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    phi0 = parameters["phi0"]
    phi1 = parameters["phi1"]
    
    # Generate consumption data using the parameters
    df["consumption_kWh"] = A0 + A1 * (np.sin((df.index-phi0)* 2*np.pi/12) + 1) + A2 * (np.sin((df.index-phi1)* 2*np.pi/24) + 1)
    dataframe["gen_consumption"] = df
    
    return dataframe

## example usage
if __name__ == "__main__":
    parameters = {"A0" : 1, "A1": 3, "A2": 2, "phi0": 3, "phi1": 9}  
    consumption_data = generate_consumption_data(parameters = parameters)["gen_consumption"]
    print(consumption_data)
    # graph the data
    plt.plot(consumption_data["date"], consumption_data["consumption_kWh"])
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.title("Consumption data")
    plt.show()