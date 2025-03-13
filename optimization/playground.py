import cvxpy as cp
import math
import numpy as np


# Function to play around with a universal optimization problem creator 
def play_around_with_optimization_problem():
    # Define the decision variables
    x = cp.Variable()
    y = cp.Variable()
    nodes = cp.Variable(5)
    cons_schedule = cp.Parameter(5)
    production_schedule = cp.Variable(5)
    
    # Define the constraints
    constraints = [
        x >= 1,
        y >= -1,
        0 == cp.sum([x, y])
    ]

    # Define the objective function
    obj = cp.Maximize(x)

    # Create the optimization problem
    prob = cp.Problem(obj, constraints)

    # Solve the problem
    prob.solve()

    # Print the results
    print(f"Optimal value of x: {x.value}")
    print(f"Optimal value of y: {y.value}")
    print(f"Optimal value of the objective function: {prob.value}")

# main 
if __name__ == "__main__":
    play_around_with_optimization_problem()