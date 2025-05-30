#!/usr/bin/env python3
"""
Minimal Dataflow Benchmark

This script provides a minimal example of using the parallel dataflow capabilities
in the PSA tool, focused on demonstrating the performance benefits with
simpler processing tasks.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import PSA tool components
from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
from src.dataflow.dataflow_classes import DataProcessingNode
from src.optimization.solver_classes import Node

# Define our own serial dataflow implementation
class SimpleDataflow:
    def __init__(self):
        self.nodes = {}
        self.dependencies = {}
        self.final_node = None
    
    def add_node(self, node, dependencies=None, final=False):
        if dependencies is None:
            dependencies = []
            
        self.nodes[node.name] = node
        self.dependencies[node.name] = dependencies
        
        if final:
            self.final_node = node
    
    def execute(self, dfs, parameters):
        # Simple topological sort execution
        processed_nodes = {}  # Store the actual result dictionaries
        
        def process_node(node_name):
            if node_name in processed_nodes:
                return processed_nodes[node_name]
            
            node = self.nodes[node_name]
            deps = self.dependencies[node_name]
            
            # Process dependencies first
            dep_results = {}
            for dep in deps:
                if dep in self.nodes:
                    dep_results[dep] = process_node(dep)
            
            # Execute this node with the actual dependency results
            result = node.process_func(dep_results, parameters)
            processed_nodes[node_name] = result
            return result
        
        # Process the final node which will recursively process all dependencies
        if self.final_node:
            final_result = process_node(self.final_node.name)
            return final_result
        return None

# Define a simple Node subclass for the benchmark
class MinimalNode(Node):
    def __init__(self, name):
        super().__init__(name)
    
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

# Define processing functions for each dataflow node
def load_data(dfs, parameters):
    """Load data from parameters or generate synthetic data"""
    print("Loading data...")
    rows = parameters.get('rows', 1000)
    cols = parameters.get('cols', 10)
    
    # Generate synthetic data
    data = pd.DataFrame(
        np.random.rand(rows, cols),
        columns=[f'col_{i}' for i in range(cols)]
    )
    
    # Add some datetime information
    data['date'] = pd.date_range(start='2024-01-01', periods=rows)
    
    # Add a target variable
    data['target'] = data.iloc[:, :cols].sum(axis=1) + np.random.normal(0, 0.1, size=rows)
    
    # Debug what's being returned
    print(f"Load returning data with shape: {data.shape}")
    
    # In parallel dataflow mode, each node's output should have just one key
    # matching the node name for consistency
    return {'data': data}

def transform_data(dfs, parameters):
    """Transform the data with some basic operations"""
    print("Transforming data...")
    
    # Extract data from input - handling both serial and parallel dataflow formats
    data = None
    try:
        # For debugging: print the keys and structure of the input
        print(f"Transform input keys: {list(dfs.keys()) if dfs else 'None'}")
        
        # Handle different possible input structures:
        
        # Case 1: Serial format - dfs has a 'load' key with nested dict
        if dfs and 'load' in dfs:
            print(f"Load node result type: {type(dfs['load'])}")
            if isinstance(dfs['load'], dict) and 'data' in dfs['load']:
                data = dfs['load']['data'].copy()
                print("Found data in dfs['load']['data']")
            elif isinstance(dfs['load'], pd.DataFrame):
                data = dfs['load'].copy()
                print("Found DataFrame directly in dfs['load']")
        
        # Case 2: Parallel format - direct access to 'data'
        elif dfs and 'data' in dfs:
            if isinstance(dfs['data'], pd.DataFrame):
                data = dfs['data'].copy()
                print("Found DataFrame directly in dfs['data']")
            else:
                print(f"dfs['data'] exists but is not a DataFrame: {type(dfs['data'])}")
                
        # Case 3: Try to find any DataFrame in the dict
        else:
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame):
                    data = value.copy()
                    print(f"Found DataFrame at key '{key}'")
                    break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame):
                            data = sub_value.copy()
                            print(f"Found DataFrame at key '{key}.{sub_key}'")
                            break
    except Exception as e:
        print(f"Error accessing data in transform: {e}")
        import traceback
        traceback.print_exc()
    
    if data is None:
        print("Warning: No data found in inputs")
        # Return an empty dataframe with the structure expected
        return {'transformed_data': pd.DataFrame({'empty': [True]})}
    
    # Just to verify what we got
    print(f"Processing data with shape: {data.shape}")
    
    try:
        # Add some derived features
        data['dayofweek'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        
        # Create some lag features
        for i in range(1, 4):
            data[f'target_lag_{i}'] = data['target'].shift(i)
        
        # Create rolling statistics
        window_sizes = [3, 7, 14]
        for window in window_sizes:
            data[f'rolling_mean_{window}'] = data['target'].rolling(window=window).mean()
            data[f'rolling_std_{window}'] = data['target'].rolling(window=window).std()
        
        # Drop rows with NaN values
        data = data.dropna()
        
        print(f"Transformed data shape: {data.shape}")
        
    except Exception as e:
        print(f"Error processing data in transform: {e}")
        import traceback
        traceback.print_exc()
        return {'transformed_data': pd.DataFrame({'error': [str(e)]})}
    
    return {'transformed_data': data}

def aggregate_results(dfs, parameters):
    """Final aggregation and output preparation"""
    print("Aggregating results...")
    
    # Print input keys for debugging
    print(f"Aggregate input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    # Get the transformed data with compatibility for both serial and parallel modes
    data = None
    try:
        # Case 1: Serial format - nested dict structure
        if 'transform' in dfs and 'transformed_data' in dfs['transform']:
            data = dfs['transform']['transformed_data']
            print("Found data in dfs['transform']['transformed_data']")
            
        # Case 2: transform is directly a DataFrame
        elif 'transform' in dfs and isinstance(dfs['transform'], pd.DataFrame):
            data = dfs['transform']
            print("Found DataFrame directly in dfs['transform']")
            
        # Case 3: Direct access to transformed_data
        elif 'transformed_data' in dfs:
            data = dfs['transformed_data']
            print("Found DataFrame directly in dfs['transformed_data']")
            
        # Case 4: Try to find any proper DataFrame in the dfs
        else:
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame):
                    if 'target' in value.columns:
                        data = value
                        print(f"Found suitable DataFrame at key '{key}'")
                        break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame) and 'target' in sub_value.columns:
                            data = sub_value
                            print(f"Found suitable DataFrame at key '{key}.{sub_key}'")
                            break
    except Exception as e:
        print(f"Error accessing data in aggregate: {e}")
        import traceback
        traceback.print_exc()
    
    if data is None:
        print("Warning: Could not find transformed data in inputs")
        # Return an empty DataFrame with the expected structure 
        return {'results': pd.DataFrame({'empty': [True], 'rows': [0], 'avg_target': [0], 
                                         'min_target': [0], 'max_target': [0], 'std_target': [0]})}
    
    print(f"Aggregating data with shape: {data.shape}")
    
    # Perform some simple aggregation
    try:
        aggs = {
            'rows': len(data),
            'avg_target': data['target'].mean(),
            'min_target': data['target'].min(),
            'max_target': data['target'].max(),
            'std_target': data['target'].std()
        }
        
        # Create a result DataFrame that can be returned
        result_df = pd.DataFrame([aggs])
        print(f"Created result DataFrame with shape: {result_df.shape}")
        
    except Exception as e:
        print(f"Error calculating aggregates: {e}")
        import traceback
        traceback.print_exc()
        return {'results': pd.DataFrame({'error': [str(e)]})}
    
    return {'results': result_df}  # Note the 'results' key which is required

# Benchmark the serial dataflow execution
def run_serial_benchmark(parameters):
    """Run a benchmark with serial dataflow execution"""
    print("\nRunning serial dataflow benchmark...")
    start_time = time.time()
    
    # Create a serial dataflow
    dataflow = SimpleDataflow()
    
    # Create and add nodes
    load_node = DataProcessingNode('load', load_data)
    transform_node = DataProcessingNode('transform', transform_data)
    result_node = DataProcessingNode('aggregate', aggregate_results)
    
    # Add nodes to dataflow with dependencies
    dataflow.add_node(load_node)
    dataflow.add_node(transform_node, ['load'])
    dataflow.add_node(result_node, ['transform'], final=True)
    
    # Execute dataflow
    result = dataflow.execute(None, parameters)
    
    duration = time.time() - start_time
    print(f"Serial dataflow completed in {duration:.2f} seconds")
    
    return result, duration

# Benchmark the parallel dataflow execution
def run_parallel_benchmark(parameters):
    """Run a benchmark with parallel dataflow execution"""
    print("\nRunning parallel dataflow benchmark...")
    
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        start_time = time.time()
        
        # Create dataflow
        dataflow = manager.newDataFlow(MinimalNode)
        dataflow._final_df_name = 'results'  # This tells the manager which key to look for
        
        # Add nodes to the dataflow
        load_node = dataflow.node("load", DataProcessingNode)
        load_node.process_func = load_data
        
        transform_node = dataflow.node("transform", DataProcessingNode)
        transform_node.process_func = transform_data
        # IMPORTANT: The inputs attribute is used to determine predecessors
        transform_node.inputs = ["load"]
        
        aggregate_node = dataflow.node("aggregate", DataProcessingNode, final=True)
        aggregate_node.process_func = aggregate_results
        aggregate_node.inputs = ["transform"]
        
        # Setup dependencies manually to ensure they're processed
        transform_node._dependencies.append(load_node)
        aggregate_node._dependencies.append(transform_node)
        
        # Execute dataflow
        task_id = manager.executeDataFlow(MinimalNode, parameters)
        
        # Wait for result
        result = manager.getData(task_id, wait=True)
        
        duration = time.time() - start_time
        print(f"Parallel dataflow completed in {duration:.2f} seconds")
        
        # Convert the result to a format that matches the serial benchmark
        result_dict = {}
        if isinstance(result, np.ndarray):
            # The manager flattened the result - convert back to more usable format
            result_dict = {'results': pd.DataFrame([{'value': float(x) for x in result}])}
        else:
            # Got a direct value back
            result_dict = {'results': result}
        
        return result_dict, duration
        
    except Exception as e:
        print(f"Error in parallel benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None, float('inf')

def run_benchmarks():
    """Run benchmarks with different data sizes"""
    results = {
        'small': {},
        'medium': {},
        'large': {}
    }
    
    # Create output directory if needed
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        'benchmark_results'
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        # Small dataset
        small_params = {'rows': 1000, 'cols': 5}
        print(f"\n=== Running benchmark with small dataset ({small_params['rows']} rows) ===")
        
        small_serial_result, small_serial_time = run_serial_benchmark(small_params)
        results['small']['Serial'] = small_serial_time
        
        small_parallel_result, small_parallel_time = run_parallel_benchmark(small_params)
        results['small']['Parallel'] = small_parallel_time
        
        # Medium dataset
        medium_params = {'rows': 5000, 'cols': 10}
        print(f"\n=== Running benchmark with medium dataset ({medium_params['rows']} rows) ===")
        
        medium_serial_result, medium_serial_time = run_serial_benchmark(medium_params)
        results['medium']['Serial'] = medium_serial_time
        
        medium_parallel_result, medium_parallel_time = run_parallel_benchmark(medium_params)
        results['medium']['Parallel'] = medium_parallel_time
        
        # Large dataset
        large_params = {'rows': 20000, 'cols': 15}
        print(f"\n=== Running benchmark with large dataset ({large_params['rows']} rows) ===")
        
        large_serial_result, large_serial_time = run_serial_benchmark(large_params)
        results['large']['Serial'] = large_serial_time
        
        large_parallel_result, large_parallel_time = run_parallel_benchmark(large_params)
        results['large']['Parallel'] = large_parallel_time
        
    finally:
        # Ensure manager is properly shut down
        if manager:
            manager.shutdown()
    
    # Calculate speedups
    for dataset_size in results:
        serial_time = results[dataset_size]['Serial']
        parallel_time = results[dataset_size]['Parallel']
        
        if parallel_time > 0:
            speedup = serial_time / parallel_time
        else:
            speedup = 0
            
        results[dataset_size]['Speedup'] = speedup
    
    # Generate plot
    plt.figure(figsize=(12, 8))
    
    # Plot execution times
    plt.subplot(2, 1, 1)
    dataset_sizes = list(results.keys())
    x = range(len(dataset_sizes))
    
    serial_times = [results[size]['Serial'] for size in dataset_sizes]
    parallel_times = [results[size]['Parallel'] for size in dataset_sizes]
    
    bar_width = 0.35
    plt.bar([i - bar_width/2 for i in x], serial_times, width=bar_width, label='Serial')
    plt.bar([i + bar_width/2 for i in x], parallel_times, width=bar_width, label='Parallel')
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Dataflow Execution Time by Dataset Size')
    plt.xticks(x, dataset_sizes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add time labels
    for i, v in enumerate(serial_times):
        plt.text(i - bar_width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    for i, v in enumerate(parallel_times):
        plt.text(i + bar_width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    # Plot speedups
    plt.subplot(2, 1, 2)
    speedups = [results[size]['Speedup'] for size in dataset_sizes]
    
    plt.bar(x, speedups, color='green')
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='No speedup')
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Speedup (x times faster)')
    plt.title('Parallel Dataflow Speedup Compared to Serial Execution')
    plt.xticks(x, dataset_sizes)
    plt.grid(axis='y', alpha=0.3)
    
    # Add speedup labels
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'minimal_dataflow_benchmark_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save results to CSV
    data = []
    for dataset_size in results:
        data.append([
            dataset_size, 
            'Serial', 
            results[dataset_size]['Serial'],
            1.0
        ])
        data.append([
            dataset_size, 
            'Parallel', 
            results[dataset_size]['Parallel'],
            results[dataset_size]['Speedup']
        ])
    
    df = pd.DataFrame(data, columns=['Dataset', 'Method', 'Time (s)', 'Speedup'])
    csv_path = os.path.join(output_dir, f'minimal_dataflow_benchmark_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for dataset_size in results:
        print(f"\n{dataset_size.upper()} dataset:")
        for method in ['Serial', 'Parallel']:
            if method == 'Serial':
                print(f"  {method}: {results[dataset_size][method]:.2f}s")
            else:
                print(f"  {method}: {results[dataset_size][method]:.2f}s (Speedup: {results[dataset_size]['Speedup']:.2f}x)")
    
    print(f"\nResults saved to {csv_path}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    try:
        run_benchmarks()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
