#!/usr/bin/env python3
"""
Dataflow Benchmark Script: Compares in-memory vs. DuckDB dataflow implementations
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import threading
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node

# Define a simple Node subclass for demonstration
class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

class DataflowBenchmark:
    """
    A utility class for benchmarking different dataflow implementations.
    This allows for direct comparison between in-memory and DuckDB-based dataflows.
    """
    
    def __init__(self, benchmark_name, output_dir='benchmark_results'):
        """Initialize benchmark with name and output directory"""
        self.benchmark_name = benchmark_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.process = psutil.Process(os.getpid())
        
    def run_benchmark(self, name, setup_func, execute_func, cleanup_func=None, repeat=3):
        """Run a benchmark for a specific implementation"""
        all_times = []
        all_memory = []
        
        print(f"\n--- Running {name} benchmark ---")
        
        for i in range(repeat):
            print(f"  Run {i+1}/{repeat}...")
            
            # Record initial memory
            initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            
            # Setup
            dataflow, params = setup_func()
            
            # Record pre-execution memory
            pre_exec_memory = self.process.memory_info().rss / (1024 * 1024)
            setup_memory = pre_exec_memory - initial_memory
            
            # Execute and time
            start_time = time.time()
            result = execute_func(dataflow, params)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record post-execution memory
            post_exec_memory = self.process.memory_info().rss / (1024 * 1024)
            execution_memory = post_exec_memory - pre_exec_memory
            total_memory = post_exec_memory - initial_memory
            
            all_times.append(execution_time)
            all_memory.append(total_memory)
            
            print(f"    Time: {execution_time:.2f}s, Memory: {total_memory:.2f}MB")
            
            # Cleanup
            if cleanup_func:
                cleanup_func(dataflow)
        
        # Calculate average metrics
        avg_time = sum(all_times) / len(all_times)
        avg_memory = sum(all_memory) / len(all_memory)
        
        # Store results
        result_data = {
            "name": name,
            "avg_execution_time": avg_time,
            "avg_memory_usage": avg_memory,
            "all_execution_times": all_times,
            "all_memory_usages": all_memory
        }
        
        self.results.append(result_data)
        
        print(f"  Average Time: {avg_time:.2f}s, Average Memory: {avg_memory:.2f}MB")
        
        return result_data
    
    def compare_results(self):
        """Compare benchmark results and return as a DataFrame"""
        if len(self.results) < 2:
            print("Warning: Need at least 2 benchmark results to compare.")
            return pd.DataFrame(self.results)
            
        # Create comparison dataframe
        df = pd.DataFrame(self.results)
        
        # Calculate relative metrics using the first result as baseline
        baseline_time = df.iloc[0]["avg_execution_time"]
        baseline_memory = df.iloc[0]["avg_memory_usage"]
        
        df["time_ratio"] = df["avg_execution_time"] / baseline_time
        df["memory_ratio"] = df["avg_memory_usage"] / baseline_memory
        
        return df
    
    def generate_plots(self, save=True):
        """Generate plots comparing the benchmark results"""
        if len(self.results) < 1:
            print("Warning: No benchmark results to plot.")
            return None, None
        
        # Extract data
        names = [r["name"] for r in self.results]
        times = [r["avg_execution_time"] for r in self.results]
        memories = [r["avg_memory_usage"] for r in self.results]
        
        # Set up figures
        time_fig, time_ax = plt.subplots(figsize=(10, 6))
        mem_fig, mem_ax = plt.subplots(figsize=(10, 6))
        
        # Time comparison
        bars = time_ax.bar(names, times, color='skyblue')
        time_ax.set_ylabel('Execution Time (seconds)')
        time_ax.set_title(f'{self.benchmark_name} - Execution Time Comparison')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            time_ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}s', ha='center', va='bottom')
        
        # Memory comparison
        bars = mem_ax.bar(names, memories, color='lightgreen')
        mem_ax.set_ylabel('Memory Usage (MB)')
        mem_ax.set_title(f'{self.benchmark_name} - Memory Usage Comparison')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            mem_ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}MB', ha='center', va='bottom')
        
        # Save plots if requested
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            time_fig.savefig(os.path.join(self.output_dir, 
                                          f'{self.benchmark_name}_time_{timestamp}.png'))
            mem_fig.savefig(os.path.join(self.output_dir, 
                                         f'{self.benchmark_name}_memory_{timestamp}.png'))
        
        return time_fig, mem_fig
    
    def save_results(self):
        """Save benchmark results to a CSV file"""
        if not self.results:
            print("Warning: No benchmark results to save.")
            return None
            
        # Convert to DataFrame
        df = self.compare_results()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{self.benchmark_name}_results_{timestamp}.csv'
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Benchmark results saved to: {filepath}")
        
        return filepath

# Define processing functions with varying data sizes for benchmarking
def create_processing_functions(data_size=10000):
    """Create processing functions with configurable data sizes for benchmarking."""
    
    def processing_a(dfs, parameters):
        """Processing function that generates a large dataframe."""
        print(f"[{threading.current_thread().name}] Processing A: Generating {data_size} rows")
        
        # Create timestamps day by day to avoid overflow
        timestamps = [pd.Timestamp('2023-01-01') + pd.Timedelta(days=i % 365) for i in range(data_size)]
        
        # Generate a large dataframe
        result_df = pd.DataFrame({
            'timestamp': timestamps,
            'values': np.random.rand(data_size) * parameters.get('factor_a', 1.0),
            'category': np.random.choice(['A', 'B', 'C'], size=data_size)
        })
        
        print(f"[{threading.current_thread().name}] Processing A complete")
        return {'results': result_df}

    def processing_b(dfs, parameters):
        """Processing function that filters data from node A."""
        print(f"[{threading.current_thread().name}] Processing B: Filtering data from node A")
        
        # Use A's output if available
        input_df = None
        for key, df in dfs.items():
            input_df = df
            break
            
        if input_df is None or input_df.empty:
            print("Warning: No input data for processing B")
            return {'results': pd.DataFrame()}
            
        # Filter to category 'A' only
        result_df = input_df[input_df['category'] == 'A'].copy()
        result_df['values'] = result_df['values'] * parameters.get('factor_b', 2.0)
        
        print(f"[{threading.current_thread().name}] Processing B complete with {len(result_df)} rows")
        return {'filtered_results': result_df}

    def processing_c(dfs, parameters):
        """Processing function that aggregates data from node B."""
        print(f"[{threading.current_thread().name}] Processing C: Aggregating data")
        
        # Use B's output if available
        input_df = None
        for key, df in dfs.items():
            input_df = df
            break
            
        if input_df is None or input_df.empty:
            print("Warning: No input data for processing C")
            return {'results': pd.DataFrame()}
        
        # Aggregate by date
        result_df = input_df.resample('D', on='timestamp').agg({'values': 'sum'}).reset_index()
        
        print(f"[{threading.current_thread().name}] Processing C complete with {len(result_df)} rows")
        return {'aggregated_results': result_df}

    def processing_final(dfs, parameters):
        """Final processing function that combines results from C."""
        print(f"[{threading.current_thread().name}] Processing FINAL: Combining all results")
        
        # Collect all input dataframes
        if not dfs:
            print("Warning: No input data for final processing")
            return {'results': pd.DataFrame()}
        
        # Combine metadata from all inputs
        stats = pd.DataFrame([
            {'source': key, 'row_count': len(df), 'sum_values': df['values'].sum() if 'values' in df.columns else 0}
            for key, df in dfs.items() if not df.empty
        ])
        
        print(f"[{threading.current_thread().name}] Final processing complete")
        return {'summary_results': stats}
        
    return {
        'processing_a': processing_a,
        'processing_b': processing_b,
        'processing_c': processing_c,
        'processing_final': processing_final
    }

# Setup functions for the benchmark

def setup_inmemory_dataflow(data_size):
    """Setup function for in-memory dataflow benchmark."""
    
    def setup():
        # Get processing functions with specified data size
        funcs = create_processing_functions(data_size)
        
        # Get manager instance
        manager = ParallelExecutionDataFlowManager.getInstance()
        
        # Create a dataflow
        dataflow = manager.newDataFlow(DemoNode)
        
        # Create nodes
        node_a = dataflow.node("A", ParallelExecutionNode)
        node_a.process_func = funcs['processing_a']
        
        node_b = dataflow.node("B", ParallelExecutionNode)
        node_b.process_func = funcs['processing_b']
        node_b.add_dependency(node_a)
        
        node_c = dataflow.node("C", ParallelExecutionNode, final=True)
        node_c.process_func = funcs['processing_c']
        node_c.add_dependency(node_b)
        
        # Parameters for the execution
        params = {
            'factor_a': 2.0,
            'factor_b': 3.0,
        }
        
        return dataflow, params
    
    return setup

def setup_duckdb_dataflow(data_size):
    """Setup function for DuckDB dataflow benchmark."""
    
    def setup():
        # Get processing functions with specified data size
        funcs = create_processing_functions(data_size)
        
        # Get manager instance
        manager = ParallelExecutionDataFlowManager.getInstance()
        
        # Create a dataflow
        dataflow = manager.newDataFlow(DemoNode)
        
        # Wrap with DuckDB persistence
        duckdb_dataflow = DuckDBParallelDataflow(dataflow)
        
        # Create nodes with DuckDB persistence
        node_a = dataflow.node("A", DuckDBParallelExecutionNode)
        node_a.process_func = funcs['processing_a']
        
        node_b = dataflow.node("B", DuckDBParallelExecutionNode)
        node_b.process_func = funcs['processing_b']
        node_b.add_dependency(node_a)
        
        node_c = dataflow.node("C", DuckDBParallelExecutionNode, final=True)
        node_c.process_func = funcs['processing_c']
        node_c.add_dependency(node_b)
        
        # Parameters for the execution
        params = {
            'factor_a': 2.0,
            'factor_b': 3.0,
        }
        
        return duckdb_dataflow, params
    
    return setup

# Execution and cleanup functions

def execute_inmemory(dataflow, params):
    """Execute function for in-memory dataflow benchmark."""
    return dataflow.execute(params)

def execute_duckdb(dataflow, params):
    """Execute function for DuckDB dataflow benchmark."""
    return dataflow.execute(params)

def cleanup_inmemory(dataflow):
    """Cleanup function for in-memory dataflow benchmark."""
    manager = ParallelExecutionDataFlowManager.getInstance()
    # Don't shutdown here, as we need to reuse the manager
    
def cleanup_duckdb(dataflow):
    """Cleanup function for DuckDB dataflow benchmark."""
    # Clear the execution data from DuckDB
    dataflow.clear_execution_data()
    # Don't shutdown here, as we need to reuse the manager

def run_benchmark(data_sizes=None, repeat=3):
    """Run benchmarks comparing in-memory and DuckDB dataflows with different data sizes."""
    
    if data_sizes is None:
        data_sizes = [10000, 50000, 100000]
    
    for size in data_sizes:
        benchmark_name = f"Dataflow_Benchmark_Size_{size}"
        print(f"\n=== Running benchmark with data size: {size} ===")
        
        # Create benchmark
        benchmark = DataflowBenchmark(benchmark_name)
        
        # Run in-memory benchmark
        benchmark.run_benchmark(
            name=f"In-Memory_{size}",
            setup_func=setup_inmemory_dataflow(size),
            execute_func=execute_inmemory,
            cleanup_func=cleanup_inmemory,
            repeat=repeat
        )
        
        # Run DuckDB benchmark
        benchmark.run_benchmark(
            name=f"DuckDB_{size}",
            setup_func=setup_duckdb_dataflow(size),
            execute_func=execute_duckdb,
            cleanup_func=cleanup_duckdb,
            repeat=repeat
        )
        
        # Generate and display comparison
        comparison = benchmark.compare_results()
        print("\nComparison Results:")
        print(comparison)
        
        # Generate plots
        benchmark.generate_plots()
        
        # Save results
        benchmark.save_results()
    
    # Final cleanup
    manager = ParallelExecutionDataFlowManager.getInstance()
    manager.shutdown()

if __name__ == "__main__":
    # For testing, use smaller data sizes first
    # You can increase these values for more comprehensive benchmarks
    data_sizes = [1000, 5000]  # Smaller data sizes for quick testing
    run_benchmark(data_sizes=data_sizes, repeat=2)
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved in benchmark_results directory")
