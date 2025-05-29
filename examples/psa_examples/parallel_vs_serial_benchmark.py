#!/usr/bin/env python3
"""
Parallel vs Serial Execution Benchmark

This script compares the performance of:
1. Fully serial execution
2. Parallel execution nodes
3. Parallel dataflows
4. Combined parallel execution nodes and dataflows

It measures execution time for CPU-bound and I/O-bound tasks to demonstrate
the performance benefits of parallel processing in different scenarios.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_root)

try:
    from src.dataflow.parallel_execution_dataflow import ParallelExecutionNode
except ImportError:
    from dataflow.parallel_execution_dataflow import ParallelExecutionNode

try:
    from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
except ImportError:
    from dataflow.dataflow_manager_paralell import ParallelDataFlowManager

try:
    from src.dataflow.parallel_dataflow_classes import ParallelProcessingNode
except ImportError:
    from dataflow.parallel_dataflow_classes import ParallelProcessingNode

try:
    from src.dataflow.dataflow_classes import SerialDataFlow, DataProcessingNode
except ImportError:
    from dataflow.dataflow_classes import SerialDataFlow, DataProcessingNode

try:
    from src.optimization.solver_classes import Node
except ImportError:
    from optimization.solver_classes import Node

print(f"Using project root path: {project_root}")

# Define a simple Node subclass for demonstration
class BenchmarkNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

# Define various types of processing workloads
def cpu_intensive_task(size):
    """A CPU-bound task that performs matrix multiplication"""
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    return np.dot(matrix_a, matrix_b)

def io_intensive_task(delay):
    """An I/O-bound task that simulates I/O operations with sleep"""
    time.sleep(delay)
    return pd.DataFrame({'data': np.random.rand(100)})

# Processing functions for dataflow nodes
def cpu_bound_processing(dfs, parameters):
    """CPU-bound processing function for dataflow nodes"""
    print(f"Starting CPU-bound process with parameters: {parameters}")
    size = parameters.get('matrix_size', 200)
    iterations = parameters.get('iterations', 3)
    
    results = []
    for i in range(iterations):
        result = cpu_intensive_task(size)
        results.append(np.mean(result))
    
    result_df = pd.DataFrame({
        'iteration': list(range(iterations)),
        'result': results
    })
    
    print("CPU-bound process completed")
    return {'results': result_df}

def io_bound_processing(dfs, parameters):
    """I/O-bound processing function for dataflow nodes"""
    print(f"Starting I/O-bound process with parameters: {parameters}")
    delay = parameters.get('delay', 1.0)
    iterations = parameters.get('iterations', 3)
    
    results = []
    for i in range(iterations):
        result = io_intensive_task(delay)
        results.append(np.mean(result['data']))
    
    result_df = pd.DataFrame({
        'iteration': list(range(iterations)),
        'result': results
    })
    
    print("I/O-bound process completed")
    return {'results': result_df}

# Serial execution functions (for comparison)
def run_serial_cpu_tasks(parameters_list):
    """Run CPU-bound tasks in serial"""
    results = []
    for params in parameters_list:
        result = cpu_bound_processing({}, params)
        results.append(result['results'])
    return pd.concat(results, ignore_index=True)

def run_serial_io_tasks(parameters_list):
    """Run I/O-bound tasks in serial"""
    results = []
    for params in parameters_list:
        result = io_bound_processing({}, params)
        results.append(result['results'])
    return pd.concat(results, ignore_index=True)

# Run benchmark with serial dataflow
def run_serial_dataflow_benchmark(task_type, parameters_list):
    """Run benchmark with serial dataflow"""
    print(f"\nRunning serial dataflow benchmark for {task_type} tasks...")
    
    start_time = time.time()
    
    # Create parameters list with task indices
    for i, params in enumerate(parameters_list):
        params['task_id'] = i
    
    # Create a serial dataflow for each parameter set
    results = []
    for params in parameters_list:
        # Create a serial dataflow
        dataflow = SerialDataFlow()
        
        # Add a node to the dataflow
        if task_type == 'cpu':
            node = DataProcessingNode('task_processor', cpu_bound_processing)
        else:
            node = DataProcessingNode('task_processor', io_bound_processing)
        
        dataflow.add_node(node, [], final=True)
        
        # Execute the dataflow
        result = dataflow.execute({}, params)
        results.append(result['results'])
    
    duration = time.time() - start_time
    print(f"Serial dataflow completed in {duration:.2f} seconds")
    
    return pd.concat(results, ignore_index=True), duration

# Run benchmark with parallel execution nodes
def run_parallel_node_benchmark(task_type, parameters_list):
    """Run benchmark with parallel execution nodes in a serial dataflow"""
    print(f"\nRunning parallel node benchmark for {task_type} tasks...")
    
    start_time = time.time()
    
    # Create parameters list with task indices
    for i, params in enumerate(parameters_list):
        params['task_id'] = i
    
    # Create a serial dataflow for each parameter set
    results = []
    for params in parameters_list:
        # Create a serial dataflow with parallel execution node
        dataflow = SerialDataFlow()
        
        # Add a parallel execution node to the dataflow
        if task_type == 'cpu':
            node = ParallelExecutionNode('task_processor', cpu_bound_processing)
        else:
            node = ParallelExecutionNode('task_processor', io_bound_processing)
        
        dataflow.add_node(node, [], final=True)
        
        # Execute the dataflow
        result = dataflow.execute({}, params)
        results.append(result['results'])
    
    duration = time.time() - start_time
    print(f"Parallel node benchmark completed in {duration:.2f} seconds")
    
    return pd.concat(results, ignore_index=True), duration

# Run benchmark with parallel dataflow
def run_parallel_dataflow_benchmark(task_type, parameters_list):
    """Run benchmark with parallel dataflow"""
    print(f"\nRunning parallel dataflow benchmark for {task_type} tasks...")
    
    manager = ParallelDataFlowManager.getInstance()
    
    start_time = time.time()
    
    # Create a dataflow for the task type
    dataflow = manager.newDataFlow(BenchmarkNode)
    
    # Set up the dataflow with appropriate processing function
    if task_type == 'cpu':
        node = dataflow.node("cpu_processor", ParallelExecutionNode, final=True)
        node.process_func = cpu_bound_processing
    else:
        node = dataflow.node("io_processor", ParallelExecutionNode, final=True)
        node.process_func = io_bound_processing
    
    # Execute the dataflow for each parameter set
    task_ids = []
    for params in parameters_list:
        task_id = manager.executeDataFlow(BenchmarkNode, params)
        task_ids.append(task_id)
    
    # Wait for all tasks to complete and get results
    results = []
    for task_id in task_ids:
        result = manager.getData(task_id, wait=True)
        if result is not None:
            results.append(result)
    
    duration = time.time() - start_time
    print(f"Parallel dataflow completed in {duration:.2f} seconds")
    
    # Convert results to DataFrame for consistency
    result_df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    
    return result_df, duration

# Run benchmark with combined parallel techniques
def run_combined_parallel_benchmark(task_type, parameters_list):
    """Run benchmark with both parallel dataflow and parallel execution nodes"""
    print(f"\nRunning combined parallel benchmark for {task_type} tasks...")
    
    manager = ParallelDataFlowManager.getInstance()
    
    start_time = time.time()
    
    # Create a dataflow for the task type
    dataflow = manager.newDataFlow(BenchmarkNode)
    
    # Set up the dataflow with appropriate processing function
    if task_type == 'cpu':
        node = dataflow.node("cpu_processor", ParallelExecutionNode, final=True)
        node.process_func = cpu_bound_processing
    else:
        node = dataflow.node("io_processor", ParallelExecutionNode, final=True)
        node.process_func = io_bound_processing
    
    # Execute the dataflow for each parameter set
    task_ids = []
    for params in parameters_list:
        # Enable parallelism within the node too
        params['use_parallel'] = True
        task_id = manager.executeDataFlow(BenchmarkNode, params)
        task_ids.append(task_id)
    
    # Wait for all tasks to complete and get results
    results = []
    for task_id in task_ids:
        result = manager.getData(task_id, wait=True)
        if result is not None:
            results.append(result)
    
    duration = time.time() - start_time
    print(f"Combined parallel techniques completed in {duration:.2f} seconds")
    
    # Convert results to DataFrame for consistency
    result_df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    
    return result_df, duration

def plot_results(benchmark_results, output_path=None):
    """Plot benchmark results"""
    task_types = benchmark_results.keys()
    
    plt.figure(figsize=(12, 8))
    
    for i, task_type in enumerate(task_types):
        results = benchmark_results[task_type]
        
        plt.subplot(len(task_types), 1, i+1)
        
        techniques = list(results.keys())
        durations = [results[technique]['duration'] for technique in techniques]
        
        # Calculate speedups relative to serial execution
        serial_duration = results['Serial']['duration']
        speedups = [serial_duration / duration for duration in durations]
        
        plt.barh(techniques, speedups, color=['blue', 'green', 'orange', 'red'])
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Speedup (relative to serial execution)')
        plt.title(f'{task_type.upper()}-Bound Task Performance')
        
        # Add duration labels
        for j, (technique, duration) in enumerate(zip(techniques, durations)):
            plt.text(0.1, j, f"{duration:.2f}s", va='center', color='white', fontweight='bold')
            plt.text(speedups[j] - 0.3, j, f"{speedups[j]:.2f}x", va='center', 
                     color='black' if speedups[j] > 2 else 'white', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
    
    plt.close()

def run_benchmarks(cpu_parameters_list, io_parameters_list):
    """Run all benchmarks and collect results"""
    results = {
        'cpu': {},
        'io': {}
    }
    
    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        # CPU-bound tasks
        print("\nRunning CPU-bound benchmarks...")
        
        # Serial execution
        _, serial_cpu_duration = run_serial_dataflow_benchmark('cpu', cpu_parameters_list)
        results['cpu']['Serial'] = {'duration': serial_cpu_duration}
        
        # Parallel execution nodes
        _, parallel_node_cpu_duration = run_parallel_node_benchmark('cpu', cpu_parameters_list)
        results['cpu']['Parallel Nodes'] = {'duration': parallel_node_cpu_duration}
        
        # Parallel dataflow
        _, parallel_df_cpu_duration = run_parallel_dataflow_benchmark('cpu', cpu_parameters_list)
        results['cpu']['Parallel Dataflow'] = {'duration': parallel_df_cpu_duration}
        
        # Combined parallel techniques
        _, combined_cpu_duration = run_combined_parallel_benchmark('cpu', cpu_parameters_list)
        results['cpu']['Combined Parallel'] = {'duration': combined_cpu_duration}
        
        # I/O-bound tasks
        print("\nRunning I/O-bound benchmarks...")
        
        # Serial execution
        _, serial_io_duration = run_serial_dataflow_benchmark('io', io_parameters_list)
        results['io']['Serial'] = {'duration': serial_io_duration}
        
        # Parallel execution nodes
        _, parallel_node_io_duration = run_parallel_node_benchmark('io', io_parameters_list)
        results['io']['Parallel Nodes'] = {'duration': parallel_node_io_duration}
        
        # Parallel dataflow
        _, parallel_df_io_duration = run_parallel_dataflow_benchmark('io', io_parameters_list)
        results['io']['Parallel Dataflow'] = {'duration': parallel_df_io_duration}
        
        # Combined parallel techniques
        _, combined_io_duration = run_combined_parallel_benchmark('io', io_parameters_list)
        results['io']['Combined Parallel'] = {'duration': combined_io_duration}
    
    finally:
        # Shutdown the parallel dataflow manager
        if manager:
            manager.shutdown()
    
    # Plot results
    plot_path = os.path.join(output_dir, f'parallel_benchmark_results_{timestamp}.png')
    plot_results(results, plot_path)
    
    # Save numerical results
    results_df = pd.DataFrame({
        'Task Type': ['CPU', 'CPU', 'CPU', 'CPU', 'IO', 'IO', 'IO', 'IO'],
        'Technique': ['Serial', 'Parallel Nodes', 'Parallel Dataflow', 'Combined Parallel'] * 2,
        'Duration': [
            results['cpu']['Serial']['duration'],
            results['cpu']['Parallel Nodes']['duration'],
            results['cpu']['Parallel Dataflow']['duration'],
            results['cpu']['Combined Parallel']['duration'],
            results['io']['Serial']['duration'],
            results['io']['Parallel Nodes']['duration'],
            results['io']['Parallel Dataflow']['duration'],
            results['io']['Combined Parallel']['duration']
        ]
    })
    
    # Calculate speedups
    results_df['Speedup'] = [
        1.0,
        results['cpu']['Serial']['duration'] / results['cpu']['Parallel Nodes']['duration'],
        results['cpu']['Serial']['duration'] / results['cpu']['Parallel Dataflow']['duration'],
        results['cpu']['Serial']['duration'] / results['cpu']['Combined Parallel']['duration'],
        1.0,
        results['io']['Serial']['duration'] / results['io']['Parallel Nodes']['duration'],
        results['io']['Serial']['duration'] / results['io']['Parallel Dataflow']['duration'],
        results['io']['Serial']['duration'] / results['io']['Combined Parallel']['duration']
    ]
    
    csv_path = os.path.join(output_dir, f'parallel_benchmark_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Numerical results saved to {csv_path}")
    
    # Print a summary
    print("\n=== BENCHMARK SUMMARY ===")
    print("CPU-bound tasks:")
    for technique, data in results['cpu'].items():
        speedup = results['cpu']['Serial']['duration'] / data['duration']
        print(f"  {technique}: {data['duration']:.2f}s (Speedup: {speedup:.2f}x)")
    
    print("\nI/O-bound tasks:")
    for technique, data in results['io'].items():
        speedup = results['io']['Serial']['duration'] / data['duration']
        print(f"  {technique}: {data['duration']:.2f}s (Speedup: {speedup:.2f}x)")
    
    return results

def main():
    # Detect available CPU count for scaling the benchmark appropriately
    cpu_count = multiprocessing.cpu_count()
    print(f"Running on system with {cpu_count} logical CPU cores")
    
    # CPU-bound task parameters
    # We'll run several matrix multiplication tasks with different sizes
    cpu_parameters_list = [
        {'matrix_size': 500, 'iterations': 3},
        {'matrix_size': 600, 'iterations': 2},
        {'matrix_size': 700, 'iterations': 2},
        {'matrix_size': 800, 'iterations': 1}
    ]
    
    # I/O-bound task parameters
    # We'll run several I/O simulation tasks with different delays
    io_parameters_list = [
        {'delay': 1.0, 'iterations': 3},
        {'delay': 1.5, 'iterations': 2},
        {'delay': 2.0, 'iterations': 2},
        {'delay': 2.5, 'iterations': 1}
    ]
    
    # Run all benchmarks
    results = run_benchmarks(cpu_parameters_list, io_parameters_list)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
