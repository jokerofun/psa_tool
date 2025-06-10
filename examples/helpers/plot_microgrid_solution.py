import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.helpers.microgrid_setup import MicrogridSetup

import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def plot_microgrid_solution(setup:MicrogridSetup, solution, grid_import, total_demand, solar_prod, wind_prod, soc_values, prices=None):
    print("Total grid import (kWh):", np.sum(grid_import))
    print("Grid import per hour:", grid_import)

    # Scale up the prices (not correct, but for visualization purposes)
    if prices is not None:
        prices = np.array(prices) * 1000  # Scale prices to money/MWh

    # Plot demand and production
    plt.figure(figsize=(12, 6))
    plt.plot(total_demand[:setup.T], label="Total Demand", color='red')
    plt.plot(solar_prod[:setup.T], label="Solar Production", color='yellow')
    plt.plot(wind_prod[:setup.T], label="Wind Production", color='blue')
    total_renewable_prod = np.add(solar_prod[:setup.T], wind_prod[:setup.T])
    plt.plot(total_renewable_prod, label="Total Renewable Production", color='purple')
    # Add battery SoC to the plot
    # for i in range(setup.no_batteries):
    #     soc = np.array(soc_values[i].value).flatten()
    #     plt.plot(soc[:-1], label=f"Battery {i} SoC", linestyle="--")

    # Sum up the battery SoC values and plot the total SoC
    total_soc = np.sum([np.array(soc_values[i].value).flatten()[:-1] for i in range(setup.no_batteries)], axis=0)
    plt.plot(total_soc, label="Total Battery SoC", linestyle="--", color='black')
    if prices is not None:
        plt.plot(prices[:setup.T], label="Electricity Price", linestyle=":", color='green')
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production, and Battery SoC Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/demand_production_profiles_{setup.no_homes}.png")
    plt.show()