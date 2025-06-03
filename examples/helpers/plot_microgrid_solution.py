import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.helpers.microgrid_setup import MicrogridSetup


def plot_microgrid_solution(setup:MicrogridSetup, solution, grid_import, total_demand, solar_prod, wind_prod, soc_values):
    print("Total grid import (kWh):", np.sum(grid_import))
    print("Grid import per hour:", grid_import)

    # Plot demand and production
    plt.figure(figsize=(12, 6))
    plt.plot(total_demand[:setup.T], label="Total Demand")
    plt.plot(solar_prod[:setup.T], label="Solar Production")
    plt.plot(wind_prod[:setup.T], label="Wind Production")
    plt.plot(solar_prod[:setup.T] + wind_prod[:setup.T], label="Total Renewable Production")
    # Add battery SoC to the plot
    for i in range(setup.no_batteries):
        plt.plot(soc_values[i][:-1], label=f"Battery {i} SoC", linestyle="--")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production, and Battery SoC Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/demand_production_profiles_{solution}.png")
    plt.show()