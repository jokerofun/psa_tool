class MicrogridSetup:
    T = 24 # time segments
    no_homes = 50
    no_solar_panels = 50
    solar_capacity_home = 5 # kW
    solar_capacity_school = 25 # kW
    wind_capacity = 1000 # kW
    no_batteries = 3
    battery_capacity = 250 # kWh
    battery_power = 500 # kW
    battery_efficiency = 0.9 # in %
    battery_SoC = 0
    no_big_solar_panels = 1
    no_wind_turbines = 1
    output_path = "figures/output.txt"
    benchmark_output_path = "benchmark/results/results.txt"

