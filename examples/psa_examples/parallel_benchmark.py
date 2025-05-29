#!/usr/bin/env python3
"""
Parallel vs Serial Execution Benchmark

This example demonstrates the performance difference between:
1. Serial execution using standard DataProcessingNode
2. Parallel execution using ParallelExecutionNode
3. Parallel dataflow execution using ParallelDataFlowManager

It measures and compares the execution times for CPU-bound and I/O-bound tasks.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_root)

# Import from src or directly depending on the setup
try:
    from src.dataflow.parallel_execution_dataflow import ParallelExecutionNode
    from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
    from src.dataflow.dataflow_classes import DataProcessingNode
    from src.optimization.solver_classes import Node
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    try:
        from dataflow.parallel_execution_dataflow import ParallelExecutionNode
        from dataflow.dataflow_manager_paralell import ParallelDataFlowManager
        from dataflow.dataflow_classes import DataProcessingNode
        from optimization.solver_classes import Node
    except ImportError as e:
        print(f"Alternative import also failed: {e}")
        sys.exit(1)

print(f"Using project root: {project_root}")

# Define a node for the parallel dataflow
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

# Create a simple dataflow for serial execution
class SimpleDataflow:
    def __init__(self):
        self.final_node = None
    
    def add_node(self, node, dependencies=None, final=False):
        if final:
            self.final_node = node
    
    def execute(self, dfs, parameters):
        if self.final_node:
            return self.final_node.process_func(dfs, parameters)
        return None

# Define workload functions
def cpu_task(size):
    """CPU-intensive task: matrix multiplication"""
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    return np.dot(matrix_a, matrix_b)

def io_task(delay):
    """I/O-bound task: simulates I/O operations with sleep"""
    time.sleep(delay)
    return pd.DataFrame({'data': np.random.rand(100)})

# Processing functions for nodes
def cpu_bound_processing(dfs, parameters):
    """CPU-bound processing for benchmark"""
    print(f"Starting CPU-bound process with parameters: {parameters}")
    size = parameters.get('matrix_size', 200)
    iterations = parameters.get('iterations', 3)
    
    results = []
    for i in range(iterations):
        result = cpu_task(size)
        results.append(np.mean(result))
    
    result_df = pd.DataFrame({
        'iteration': list(range(iterations)),
        'result': results
    })
    
    print(f"CPU-bound process completed with {iterations} iterations")
    return {'results': result_df}

def io_bound_processing(dfs, parameters):
    """I/O-bound processing for benchmark"""
    print(f"Starting I/O-bound process with parameters: {parameters}")
    delay = parameters.get('delay', 1.0)
    iterations = parameters.get('iterations', 3)
    
    results = []
    for i in range(iterations):
        result = io_task(delay)
        results.append(np.mean(result['data']))
    
    result_df = pd.DataFrame({
        'iteration': list(range(iterations)),
        'result': results
    })
    
    print(f"I/O-bound process completed with {iterations} iterations")
    return {'results': result_df}

# Serial execution function
def run_serial_benchmark(task_type, params_list):
    """Run tasks in serial"""
    print(f"\nRunning serial {task_type} benchmark...")
    
    start_time = time.time()
    
    results = []
    for params in params_list:
        # Create simple dataflow
        dataflow = SimpleDataflow()
        
        # Add processing node
        if task_type == 'cpu':
            node = DataProcessingNode('processor', cpu_bound_processing)
        else:
            node = DataProcessingNode('processor', io_bound_processing)
        
        dataflow.add_node(node, final=True)
        
        # Execute
        result = dataflow.execute({}, params)
        results.append(result['results'])
    
    duration = time.time() - start_time
    print(f"Serial {task_type} benchmark completed in {duration:.2f} seconds")
    
    return duration

# Run with parallel execution nodes but serial dataflow
def run_parallel_node_benchmark(task_type, params_list):
    """Run tasks with parallel execution nodes"""
    print(f"\nRunning parallel node {task_type} benchmark...")
    
    start_time = time.time()
    
    results = []
    for params in params_list:
        # Create simple dataflow
        dataflow = SimpleDataflow()
        
        # Add parallel execution node
        if task_type == 'cpu':
            node = ParallelExecutionNode('processor')
            node.process_func = cpu_bound_processing
        else:
            node = ParallelExecutionNode('processor')
            node.process_func = io_bound_processing
        
        dataflow.add_node(node, final=True)
        
        # Execute
        result = dataflow.execute({}, params)
        results.append(result['results'])
    
    duration = time.time() - start_time
    print(f"Parallel node {task_type} benchmark completed in {duration:.2f} seconds")
    
    return duration

# Run with parallel dataflow
def run_parallel_dataflow_benchmark(task_type, params_list):
    """Run tasks with parallel dataflow"""
    print(f"\nRunning parallel dataflow {task_type} benchmark...")
    
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        start_time = time.time()
        
        # Create dataflow
        dataflow = manager.newDataFlow(BenchmarkNode)
        
        # Add node
        if task_type == 'cpu':
            node = dataflow.node("processor", ParallelExecutionNode, final=True)
            node.process_func = cpu_bound_processing
        else:
            node = dataflow.node("processor", ParallelExecutionNode, final=True)
            node.process_func = io_bound_processing
        
        # Execute in parallel
        task_ids = []
        for params in params_list:
            task_id = manager.executeDataFlow(BenchmarkNode, params)
            task_ids.append(task_id)
        
        # Wait for results
        results = []
        for task_id in task_ids:
            result = manager.getData(task_id, wait=True)
            if result is not None:
                results.append(result)
        
        duration = time.time() - start_time
        print(f"Parallel dataflow {task_type} benchmark completed in {duration:.2f} seconds")
        
        return duration
    
    except Exception as e:
        print(f"Error in parallel dataflow benchmark: {e}")
        return float('inf')

# Run with combined approach
def run_combined_parallel_benchmark(task_type, params_list):
    """Run tasks with both parallel dataflow and parallel nodes"""
    print(f"\nRunning combined parallel {task_type} benchmark...")
    
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        start_time = time.time()
        
        # Create dataflow
        dataflow = manager.newDataFlow(BenchmarkNode)
        
        # Add node with explicit parallelization
        if task_type == 'cpu':
            node = dataflow.node("processor", ParallelExecutionNode, final=True)
            node.process_func = cpu_bound_processing
        else:
            node = dataflow.node("processor", ParallelExecutionNode, final=True)
            node.process_func = io_bound_processing
        
        # Execute in parallel with configuration for both levels of parallelism
        task_ids = []
        for params in params_list:
            # Add flag to indicate using parallel execution within the task too
            params['use_parallel'] = True
            task_id = manager.executeDataFlow(BenchmarkNode, params)
            task_ids.append(task_id)
        
        # Wait for results
        results = []
        for task_id in task_ids:
            result = manager.getData(task_id, wait=True)
            if result is not None:
                results.append(result)
        
        duration = time.time() - start_time
        print(f"Combined parallel {task_type} benchmark completed in {duration:.2f} seconds")
        
        return duration
    
    except Exception as e:
        print(f"Error in combined parallel benchmark: {e}")
        return float('inf')

def run_all_benchmarks(cpu_params, io_params):
    """Run all benchmarks and collect results"""
    results = {}
    
    # Ensure output directory exists
    output_dir = os.path.join(project_root, 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize the parallel dataflow manager
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        # CPU-bound benchmarks
        print("\n=== Running CPU-bound benchmark suite ===")
        cpu_serial_time = run_serial_benchmark('cpu', cpu_params)
        cpu_parallel_node_time = run_parallel_node_benchmark('cpu', cpu_params)
        cpu_parallel_dataflow_time = run_parallel_dataflow_benchmark('cpu', cpu_params)
        cpu_combined_time = run_combined_parallel_benchmark('cpu', cpu_params)
        
        results['cpu'] = {
            'Serial': cpu_serial_time,
            'Parallel Node': cpu_parallel_node_time,
            'Parallel Dataflow': cpu_parallel_dataflow_time,
            'Combined Parallel': cpu_combined_time
        }
        
        # I/O-bound benchmarks
        print("\n=== Running I/O-bound benchmark suite ===")
        io_serial_time = run_serial_benchmark('io', io_params)
        io_parallel_node_time = run_parallel_node_benchmark('io', io_params)
        io_parallel_dataflow_time = run_parallel_dataflow_benchmark('io', io_params)
        io_combined_time = run_combined_parallel_benchmark('io', io_params)
        
        results['io'] = {
            'Serial': io_serial_time,
            'Parallel Node': io_parallel_node_time,
            'Parallel Dataflow': io_parallel_dataflow_time,
            'Combined Parallel': io_combined_time
        }
        
        # Calculate speedups
        for task_type in results:
            serial_time = results[task_type]['Serial']
            methods = list(results[task_type].keys())  # Create a copy of the keys
            for method in methods:
                if method != 'Serial':
                    duration = results[task_type][method]
                    speedup = serial_time / duration if duration > 0 else 0
                    results[task_type][f'{method} Speedup'] = speedup
    
    finally:
        # Shutdown manager
        if manager:
            manager.shutdown()
    
    # Plot results
    plot_path = os.path.join(output_dir, f'parallel_benchmark_{timestamp}.png')
    plot_results(results, plot_path)
    
    # Save numerical results to CSV
    data = []
    for task_type in results:
        for method in ['Serial', 'Parallel Node', 'Parallel Dataflow', 'Combined Parallel']:
            duration = results[task_type][method]
            speedup = results[task_type].get(f'{method} Speedup', 1.0)
            data.append([task_type.upper(), method, duration, speedup])
    
    results_df = pd.DataFrame(data, columns=['Task Type', 'Method', 'Duration (s)', 'Speedup'])
    csv_path = os.path.join(output_dir, f'parallel_benchmark_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    
    print("\n=== BENCHMARK SUMMARY ===")
    for task_type in results:
        print(f"\n{task_type.upper()}-bound tasks:")
        serial_time = results[task_type]['Serial']
        for method in ['Serial', 'Parallel Node', 'Parallel Dataflow', 'Combined Parallel']:
            duration = results[task_type][method]
            speedup = serial_time / duration if duration > 0 else 0
            print(f"  {method}: {duration:.2f}s (Speedup: {speedup:.2f}x)")
    
    print(f"\nResults saved to {csv_path}")
    print(f"Plot saved to {plot_path}")
    
    return results

def plot_results(results, output_path):
    """Generate a plot of the benchmark results"""
    plt.figure(figsize=(12, 10))
    
    # Subplot for execution times
    plt.subplot(2, 1, 1)
    
    categories = ['Serial', 'Parallel Node', 'Parallel Dataflow', 'Combined Parallel']
    cpu_times = [results['cpu'][cat] for cat in categories]
    io_times = [results['io'][cat] for cat in categories]
    
    x = range(len(categories))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], cpu_times, width, label='CPU-bound')
    plt.bar([i + width/2 for i in x], io_times, width, label='I/O-bound')
    
    plt.xlabel('Execution Method')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time by Method')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add time labels
    for i, v in enumerate(cpu_times):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}s', ha='center')
    for i, v in enumerate(io_times):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    # Subplot for speedups
    plt.subplot(2, 1, 2)
    
    categories = ['Parallel Node', 'Parallel Dataflow', 'Combined Parallel']
    cpu_speedups = [results['cpu'][f'{cat} Speedup'] for cat in categories]
    io_speedups = [results['io'][f'{cat} Speedup'] for cat in categories]
    
    x = range(len(categories))
    
    plt.bar([i - width/2 for i in x], cpu_speedups, width, label='CPU-bound')
    plt.bar([i + width/2 for i in x], io_speedups, width, label='I/O-bound')
    
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='No speedup')
    plt.xlabel('Execution Method')
    plt.ylabel('Speedup (x times faster)')
    plt.title('Performance Speedup Compared to Serial Execution')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add speedup labels
    for i, v in enumerate(cpu_speedups):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}x', ha='center')
    for i, v in enumerate(io_speedups):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Main entry point for the benchmark"""
    cpu_count = multiprocessing.cpu_count()
    print(f"System has {cpu_count} logical CPU cores")
    
    # CPU-bound task parameters
    cpu_params = [
        {'matrix_size': 500, 'iterations': 2},
        {'matrix_size': 600, 'iterations': 2},
        {'matrix_size': 700, 'iterations': 1}
    ]
    
    # I/O-bound task parameters
    io_params = [
        {'delay': 0.5, 'iterations': 3},
        {'delay': 0.7, 'iterations': 2},
        {'delay': 1.0, 'iterations': 1}
    ]
    
    print("\nStarting parallel vs. serial benchmark...")
    results = run_all_benchmarks(cpu_params, io_params)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
