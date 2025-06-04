import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.helpers.microgrid_setup import MicrogridSetup

def build_microgrid(setup:MicrogridSetup,
                    demand_file_path = "../../data/gboml_data/demand.csv",
                    solar_gen_file_path = "../../data/gboml_data/gen_solar.csv",
                    big_solar_gen_file_path = "../../data/gboml_data/gen_big_solar.csv",
                    wind_gen_file_path = "../../data/gboml_data/gen_wind.csv",
                    file_path="examples/GBOML/microgrid.txt"):
    time_horizon_template = '''
#TIMEHORIZON
T = {t};
'''

    demand_template = '''
#NODE DEMAND
#PARAMETERS
total_demand = import "{demand_csv}";
#VARIABLES
external: consumption[T];
#CONSTRAINTS
consumption[t] == total_demand[t];
'''

    solar_panel_tepmlate = '''
#NODE SOLAR_PV_{i}
#PARAMETERS
max_power_output_kW = import "{solar_csv}";
#VARIABLES
internal: c[T];
external: electricity[T];
#CONSTRAINTS
c[t] >= 0;
c[t] <= 1;
electricity[t] >= 0;
electricity[t] == c[t] * max_power_output_kW[t];
'''

    big_solar_panel_template = '''
#NODE BIG_SOLAR_PV_{i}
#PARAMETERS
max_power_output_kW = import "{big_solar_csv}";
#VARIABLES
internal: c[T];
external: electricity[T];
#CONSTRAINTS
c[t] >= 0;
c[t] <= 1;
electricity[t] >= 0;
electricity[t] == c[t] * max_power_output_kW[t];
'''

    wind_turbine_template = '''
#NODE WIND_TURBINE_{i}
#PARAMETERS
max_power_output_kW = import "{wind_csv}";
#VARIABLES
external: electricity[T];
#CONSTRAINTS
electricity[t] >= 0;
electricity[t] == max_power_output_kW[t];
'''

    battery_template = '''
#NODE BATTERY_{i}
#PARAMETERS
charging_power_kW = {charging};
discharging_power_kW = {discharging};
capacity_kWh = {capacity};
efficiency = {efficiency};
SoC = {soc};
#VARIABLES
internal: energy[T];
external: charge[T];
external: discharge[T];
#CONSTRAINTS
energy[t] >= 0;
charge[t] >= 0;
discharge[t] >= 0;
energy[t] <= capacity_kWh;
charge[t] <= charging_power_kW;
discharge[t] <= discharging_power_kW;
energy[t+1] == energy[t] + efficiency * charge[t] - discharge[t] / efficiency;
energy[0] == SoC;
'''

    metering_point_template = '''
#NODE METERING_POINT
#VARIABLES
external: power_import[T];
#CONSTRAINTS
power_import[t] >= 0;
#OBJECTIVES
min: power_import[t];
'''

    power_balance_template = '''
#HYPEREDGE POWER_BALANCE
#CONSTRAINTS
'''

    power_balance_template += "METERING_POINT.power_import[t]"
    for i in range(1, setup.no_solar_panels + 1):
        power_balance_template += f" + SOLAR_PV_{i}.electricity[t]"

    for i in range(1, setup.no_big_solar_panels + 1):
        power_balance_template += f" + BIG_SOLAR_PV_{i}.electricity[t]"
    
    for i in range(1, setup.no_wind_turbines + 1):
        power_balance_template += f" + WIND_TURBINE_{i}.electricity[t]"

    for i in range(1, setup.no_batteries + 1):
        power_balance_template += f" + BATTERY_{i}.discharge[t]"

    power_balance_template += " == DEMAND.consumption[t]"

    for i in range(1, setup.no_batteries + 1):
        power_balance_template += f" + BATTERY_{i}.charge[t]"

    power_balance_template += ";"

    with open(file_path, "w") as f:
        block = time_horizon_template.format(t=setup.T)
        f.write(block + "\n")

        block = demand_template.format(demand_csv=demand_file_path)
        f.write(block + "\n")

        for i in range(1, setup.no_solar_panels + 1):
            block = solar_panel_tepmlate.format(i=i, solar_csv=f"../../data/gboml_data/gen_solar_{i}.csv")
            f.write(block + "\n")

        for i in range(1, setup.no_big_solar_panels + 1):
            block = big_solar_panel_template.format(i=i, big_solar_csv=big_solar_gen_file_path)
            f.write(block + "\n")

        for i in range(1, setup.no_wind_turbines + 1):
            block = wind_turbine_template.format(i=i, wind_csv=wind_gen_file_path)
            f.write(block + "\n")

        for i in range(1, setup.no_batteries + 1):
            block = battery_template.format(i=i, 
                                            charging=setup.battery_power, 
                                            discharging=setup.battery_power,
                                            capacity=setup.battery_capacity,
                                            efficiency=setup.battery_efficiency,
                                            soc=setup.battery_SoC)
            f.write(block + "\n")
        
        block = metering_point_template
        f.write(block + "\n")

        block = power_balance_template.format()
        f.write(block + "\n")
