import cvxpy as cp

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