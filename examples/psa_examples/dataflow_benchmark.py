import time
import pandas as pd
import numpy as np
import sys
import os
import threading
import psutil
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node
from src.dataflow.benchmark_utils import DataflowBenchmark

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

    def processing_d(dfs, parameters):
        """Processing function that performs a complex operation on node B data."""
        print(f"[{threading.current_thread().name}] Processing D: Complex calculations")
        
        # Use B's output if available
        input_df = None
        for key, df in dfs.items():
            input_df = df
            break
            
        if input_df is None or input_df.empty:
            print("Warning: No input data for processing D")
            return {'results': pd.DataFrame()}
        
        # Create a more complex transformation - rolling window calculations
        df = input_df.copy()
        df['roll_mean'] = df['values'].rolling(window=min(30, len(df)), min_periods=1).mean()
        df['roll_std'] = df['values'].rolling(window=min(30, len(df)), min_periods=1).std()
        df['normalized'] = (df['values'] - df['roll_mean']) / df['roll_std'].replace(0, 1)
        
        print(f"[{threading.current_thread().name}] Processing D complete with {len(df)} rows")
        return {'complex_results': df}

    def processing_final(dfs, parameters):
        """Final processing function that combines results from C and D."""
        print(f"[{threading.current_thread().name}] Processing FINAL: Combining all results")
        
        # Collect all input dataframes
        if not dfs:
            print("Warning: No input data for final processing")
            return {'results': pd.DataFrame()}
        
        # Combine metadata from all inputs
        stats = pd.DataFrame([
            {'source': key, 'row_count': len(df), 'sum_values': df['values'].sum()}
            for key, df in dfs.items() if not df.empty and 'values' in df.columns
        ])
        
        print(f"[{threading.current_thread().name}] Final processing complete")
        return {'summary_results': stats}
        
    return {
        'processing_a': processing_a,
        'processing_b': processing_b,
        'processing_c': processing_c,
        'processing_d': processing_d,
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
        
        node_c = dataflow.node("C", ParallelExecutionNode)
        node_c.process_func = funcs['processing_c']
        node_c.add_dependency(node_b)
        
        node_d = dataflow.node("D", ParallelExecutionNode)
        node_d.process_func = funcs['processing_d']
        node_d.add_dependency(node_b)
        
        node_final = dataflow.node("Final", ParallelExecutionNode, final=True)
        node_final.process_func = funcs['processing_final']
        node_final.add_dependency(node_c)
        node_final.add_dependency(node_d)
        
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
        
        node_c = dataflow.node("C", DuckDBParallelExecutionNode)
        node_c.process_func = funcs['processing_c']
        node_c.add_dependency(node_b)
        
        node_d = dataflow.node("D", DuckDBParallelExecutionNode)
        node_d.process_func = funcs['processing_d']
        node_d.add_dependency(node_b)
        
        node_final = dataflow.node("Final", DuckDBParallelExecutionNode, final=True)
        node_final.process_func = funcs['processing_final']
        node_final.add_dependency(node_c)
        node_final.add_dependency(node_d)
        
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
    manager.shutdown()

def cleanup_duckdb(dataflow):
    """Cleanup function for DuckDB dataflow benchmark."""
    # Clear the execution data from DuckDB
    dataflow.clear_execution_data()
    
    # Also shutdown the manager's executor
    manager = ParallelExecutionDataFlowManager.getInstance()
    manager.shutdown()

def run_benchmark(data_sizes=None, repeat=3):
    """Run benchmarks comparing in-memory and DuckDB dataflows with different data sizes."""
    
    if data_sizes is None:
        data_sizes = [10000, 50000, 100000]
    
    all_results = []
    
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
        
        # Add to all results
        all_results.append(benchmark.results)
    
    return all_results

if __name__ == "__main__":
    # For testing, use smaller data sizes first
    # You can increase these values for more comprehensive benchmarks
    data_sizes = [1000, 5000]  # Smaller data sizes for quick testing
    results = run_benchmark(data_sizes=data_sizes, repeat=2)
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved in benchmark_results directory")
