# open scalability_results.txt and plot the results with matplotlib

import matplotlib.pyplot as plt
import numpy as np
import ast
import pandas as pd

# Function to parse the results file
def parse_scalability_results(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            # Convert the dictionary string to an actual dictionary
            try:
                data = ast.literal_eval(line.strip())
                results.append(data)
            except:
                print(f"Error parsing line: {line}")
    
    return pd.DataFrame(results)

# Parse the results
df = parse_scalability_results('scalability_results.txt')
print(df)
# Create multiple plots for different analyses
plt.figure(figsize=(18, 12))

# 1. Plot execution time vs number of homes
plt.subplot(2, 2, 1)
for mode in df['execution_mode'].unique():
    for l in df['l'].unique():
        for T in [24, 72, 168]:  # Selected time periods for clarity
            subset = df[(df['execution_mode'] == mode) & 
                         (df['l'] == l) & 
                         (df['T'] == T)]
            if not subset.empty:
                plt.plot(subset['no_homes'], subset['optimizer_time'], 
                         marker='o', label=f"{mode}, l={l}, T={T}")

plt.xlabel('Number of Households')
plt.ylabel('Optimization Time (seconds)')
plt.title('Optimization Time vs. Number of Households')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 2. Plot execution time vs time horizon (T)
plt.subplot(2, 2, 2)
for mode in df['execution_mode'].unique():
    for l in df['l'].unique():
        for homes in [10, 100, 500]:  # Selected household counts for clarity
            subset = df[(df['execution_mode'] == mode) & 
                         (df['l'] == l) & 
                         (df['no_homes'] == homes)]
            if not subset.empty:
                plt.plot(subset['T'], subset['optimizer_time'], 
                         marker='o', label=f"{mode}, l={l}, homes={homes}")

plt.xlabel('Time Horizon (T)')
plt.ylabel('Optimization Time (seconds)')
plt.title('Optimization Time vs. Time Horizon')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 3. Compare serial vs parallel execution
plt.subplot(2, 2, 3)
for l in df['l'].unique():
    for homes in [10,50,100,500]:  # Focus on middle-tier home count
        serial = df[(df['execution_mode'] == 'serial') & 
                     (df['l'] == l) & 
                     (df['no_homes'] == homes)]
        parallel = df[(df['execution_mode'] == 'parallel') & 
                       (df['l'] == l) & 
                       (df['no_homes'] == homes)]
        
        if not serial.empty and not parallel.empty:
            plt.plot(serial['T'], serial['dataflow_time'], 
                     marker='o', label=f"Serial, l={l}, homes={homes}")
            plt.plot(parallel['T'], parallel['dataflow_time'], 
                     marker='s', label=f"Parallel, l={l}, homes={homes}")

plt.xlabel('Time Horizon (T)')
plt.ylabel('Dataflow Processing Time (seconds)')
plt.title('Dataflow Processing Time: Serial vs Parallel')
plt.grid(True)
plt.legend()

# 4. Plot total problem size (variables + constraints) vs execution time
plt.subplot(2, 2, 4)
df['problem_size'] = df['variables'] + df['constraints']

for mode in df['execution_mode'].unique():
    subset = df[df['execution_mode'] == mode]
    plt.scatter(subset['problem_size'], subset['optimizer_time'], 
                alpha=0.7, label=mode)

plt.xlabel('Problem Size (Variables + Constraints)')
plt.ylabel('Optimization Time (seconds)')
plt.title('Optimization Time vs. Problem Size')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('figures/scalability/scalability_analysis.png', dpi=300)
plt.show()

# Additional analysis: create a heatmap of execution time by homes and time horizon
plt.figure(figsize=(12, 10))
for mode in df['execution_mode'].unique():
    for l in df['l'].unique():
        pivot_data = df[(df['execution_mode'] == mode) & (df['l'] == l)]
        if not pivot_data.empty:
            pivot = pivot_data.pivot_table(index='no_homes', columns='T', values='optimizer_time')
            
            plt.figure(figsize=(10, 8))
            plt.imshow(pivot, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Optimization Time (seconds)')
            plt.title(f'Optimization Time Heatmap: {mode}, l={l}')
            plt.xlabel('Time Horizon (T)')
            plt.ylabel('Number of Households')
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if not np.isnan(pivot.iloc[i, j]):
                        plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", 
                                 ha="center", va="center", color="w")
            
            plt.savefig(f'figures/scalability/heatmap_{mode}_l{l}.png', dpi=300)

print("Plots have been created and saved!")