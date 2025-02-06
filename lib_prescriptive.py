import cvxpy as cp
import math
import numpy as np

# Function to solve the battery arbitrage problem
def battery_arbitrage(prices, battery_capacity, charging_power, discharging_power, efficiency):
    T = len(prices)  # Number of time periods (hours)
    
    # Decision Variables
    charge = cp.Variable(T, nonneg=True)  # Charging power
    discharge = cp.Variable(T, nonneg=True)  # Discharging power
    state_of_charge = cp.Variable(T, nonneg=True)  # Battery state of charge
    binary = cp.Variable(T, boolean=True)  # Binary to control charge/discharge mode

    # Objective: Maximize arbitrage profit
    profit = cp.sum(prices @ discharge) - cp.sum(prices @ charge)
    objective = cp.Maximize(profit)

    # Constraints
    constraints = [
        state_of_charge[0] == 0,  # Start with empty battery
        state_of_charge[-1] == 0  # End with empty battery
    ]

    M = battery_capacity  # Big-M parameter (or max(charging_power, discharging_power))

    for t in range(T):
        if t == 0:
            constraints += [
                state_of_charge[t] == efficiency * charge[t] - (1 / efficiency) * discharge[t]
            ]
        else:
            constraints += [
                state_of_charge[t] == state_of_charge[t-1] + efficiency * charge[t] - (1 / efficiency) * discharge[t]
            ]
        
        # Power and capacity constraints
        constraints += [
            charge[t] <= charging_power,
            discharge[t] <= discharging_power,
            state_of_charge[t] <= battery_capacity,
            
            # Big-M constraints (to prevent charge & discharge at the same time)
            charge[t] <= M * binary[t],                # If binary[t] == 0, charge[t] = 0
            discharge[t] <= M * (1 - binary[t])        # If binary[t] == 1, discharge[t] = 0
        ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    print(f"Optimal Profit: {problem.value:.2f} EUR")
    
    return charge.value, discharge.value, state_of_charge.value



def battery_arbitrage_multiple(prices, num_days = 7 ,num_batteries = 3, battery_capacities = [20, 60, 300], max_charging_powers= [20, 20, 10], max_discharging_powers= [20, 20, 10], efficiencies = [0.9, 0.7, 0.45]):
    """
    Solves the battery arbitrage problem for a specified number of batteries.

    Parameters:
        prices (list): Electricity prices for each time period (length T).
        num_days (int): Number of days to consider in the optimization.
        num_batteries (int): Number of batteries to use (1, 2, or 3).
        battery_capacities (list): List of capacities for each battery (MWh).
        max_charging_powers (list): List of max charging power for each battery (MW).
        max_discharging_powers (list): List of max discharging power for each battery (MW).
        efficiencies (list): List of efficiencies for each battery (fraction, e.g., 0.9 for 90%).

    Returns:
        tuple: Optimal profit, optimal charging/discharging schedules, and SOC schedules.
    """
    prices = prices[:24*num_days]  # Limit to the first num_days days
    T = len(prices)  # Number of time periods

    # Square root the efficiency to account for both charging and discharging
    efficiencies = [math.sqrt(eff) for eff in efficiencies]

    battery_capacities = battery_capacities[:num_batteries]
    max_charging_powers = max_charging_powers[:num_batteries]
    max_discharging_powers = max_discharging_powers[:num_batteries]
    efficiencies = efficiencies[:num_batteries]

    # Decision Variables
    soc = cp.Variable((T, num_batteries))  # State of charge for each battery
    charge_power = cp.Variable((T, num_batteries), nonneg=True)  # Positive charging power
    discharge_power = cp.Variable((T, num_batteries), nonneg=True)  # Positive discharging power

    # Objective: Maximize arbitrage profit
    profit = cp.sum(cp.multiply(discharge_power - charge_power, np.array(prices).reshape(-1, 1)))  # Negative u when discharging adds revenue
    objective = cp.Maximize(profit)

    # Constraints
    constraints = []

    for i in range(num_batteries):
        # Initial and final state of charge
        # constraints += [soc[0, i] == 0, soc[-1, i] == 0]
        constraints += [soc[0, i] == 0]
        for t in range(T):
            if t == 0:
                # SOC dynamics for the first time period
                constraints += [
                    soc[t, i] == efficiencies[i] * charge_power[t, i] - (1 / efficiencies[i]) * discharge_power[t, i]
                ]
            else:
                # SOC dynamics for subsequent periods
                constraints += [
                    soc[t, i] == soc[t - 1, i] + efficiencies[i] * charge_power[t, i] - (1 / efficiencies[i]) * discharge_power[t, i]
                ]

            # Power flow constraints
            constraints += [
                charge_power[t, i] <= max_charging_powers[i],  # Charging power limit
                discharge_power[t, i] <= max_discharging_powers[i],  # Discharging power limit
            ]

            # SOC limits
            constraints += [
                soc[t, i] <= battery_capacities[i],  # Max SOC
                soc[t, i] >= 0,  # Non-negative SOC
            ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=True)

    # Extract the results
    optimal_profit = problem.value
    optimal_schedule = discharge_power.value - charge_power.value
    soc_schedule = soc.value

    return optimal_profit, optimal_schedule, soc_schedule
