#!/usr/bin/env python3
"""
Quick Memory Test: A simplified benchmark focusing specifically on memory measurement
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import gc
import tracemalloc
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node

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

def measure_peak_memory(func, *args, **kwargs):
    """
    Measure peak memory usage of a function using tracemalloc
    
    Args:
        func: Function to measure
        args, kwargs: Arguments to pass to the function
    
    Returns:
        tuple: (result of function, peak memory usage in MB)
    """
    # Force garbage collection before measurement
    gc.collect()
    
    # Start memory tracing
    tracemalloc.start()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Get memory peak
    current, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / (1024 * 1024)  # Convert to MB
    
    # Stop tracing
    tracemalloc.stop()
    
    return result, peak_mb

def create_test_df(size):
    """Create a test dataframe of specified size"""
    print(f"Creating test dataframe with {size} rows...")
    
    # Create in chunks to avoid memory issues with large datasets
    chunks = []
    chunk_size = 10000
    
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        chunk_length = end - i
        
        chunk = pd.DataFrame({
            'id': range(i, end),
            'value': np.random.rand(chunk_length),
            'category': np.random.choice(['A', 'B', 'C'], size=chunk_length)
        })
        chunks.append(chunk)
    
    # Combine chunks
    df = pd.concat(chunks)
    print(f"Test dataframe created with {len(df)} rows")
    return df

def test_inmemory_node(data_size):
    """Test memory usage of in-memory node processing"""
    print(f"\n--- Testing In-Memory Node with {data_size} rows ---")
    
    # Create test data
    test_df = create_test_df(data_size)
    
    # Set up node processing function
    def process_func(dfs, params):
        # Process the data
        filtered = test_df[test_df['category'] == 'A'].copy()
        filtered['value'] = filtered['value'] * 2
        return {'result': filtered}
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    node = dataflow.node("TestNode", ParallelExecutionNode)
    node.process_func = process_func
    
    # Measure memory during execution
    _, peak_memory = measure_peak_memory(
        lambda: dataflow.execute({})
    )
    
    # Cleanup
    manager.shutdown()
    
    print(f"Peak memory usage: {peak_memory:.2f}MB")
    return peak_memory

def test_duckdb_node(data_size):
    """Test memory usage of DuckDB node processing"""
    print(f"\n--- Testing DuckDB Node with {data_size} rows ---")
    
    # Create test data
    test_df = create_test_df(data_size)
    
    # Set up node processing function
    def process_func(dfs, params):
        # Process the data
        filtered = test_df[test_df['category'] == 'A'].copy()
        filtered['value'] = filtered['value'] * 2
        return {'result': filtered}
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    duckdb_dataflow = DuckDBParallelDataflow(dataflow)
    node = dataflow.node("TestNode", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Measure memory during execution
    _, peak_memory = measure_peak_memory(
        lambda: duckdb_dataflow.execute({})
    )
    
    # Cleanup
    duckdb_dataflow.clear_execution_data()
    manager.shutdown()
    
    print(f"Peak memory usage: {peak_memory:.2f}MB")
    return peak_memory

def run_memory_tests(data_sizes=None):
    """Run memory usage tests with multiple data sizes"""
    if data_sizes is None:
        data_sizes = [1000, 5000, 10000]
    
    inmem_results = []
    duckdb_results = []
    
    # Ensure singleton is reset between tests
    ParallelExecutionDataFlowManager._instance = None
    
    for size in data_sizes:
        # Test in-memory
        inmem_mem = test_inmemory_node(size)
        inmem_results.append((size, inmem_mem))
        
        # Reset singleton
        ParallelExecutionDataFlowManager._instance = None
        
        # Force garbage collection
        gc.collect()
        
        # Test DuckDB
        duckdb_mem = test_duckdb_node(size)
        duckdb_results.append((size, duckdb_mem))
        
        # Reset singleton
        ParallelExecutionDataFlowManager._instance = None
        
        # Force garbage collection
        gc.collect()
    
    # Create output directory
    output_dir = 'benchmark_results/memory_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    sizes = [r[0] for r in inmem_results]
    inmem_mem = [r[1] for r in inmem_results]
    duckdb_mem = [r[1] for r in duckdb_results]
    ratios = [d/i if i > 0 else 0 for d, i in zip(duckdb_mem, inmem_mem)]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, inmem_mem, 'o-', label='In-Memory')
    plt.plot(sizes, duckdb_mem, 's-', label='DuckDB')
    plt.xlabel('Data Size (rows)')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'memory_usage_{timestamp}.png'))
    
    # Plot ratio
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, ratios, 'o-')
    plt.axhline(y=1.0, color='gray', linestyle='--')
    plt.xlabel('Data Size (rows)')
    plt.ylabel('Memory Ratio (DuckDB/In-Memory)')
    plt.title('Memory Usage Ratio')
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'memory_ratio_{timestamp}.png'))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'data_size': sizes,
        'inmem_memory': inmem_mem,
        'duckdb_memory': duckdb_mem,
        'memory_ratio': ratios
    })
    
    results_df.to_csv(os.path.join(output_dir, f'memory_results_{timestamp}.csv'), index=False)
    
    # Save summary text
    with open(os.path.join(output_dir, f'memory_summary_{timestamp}.txt'), 'w') as f:
        f.write("=== Memory Usage Test Results ===\n\n")
        
        for i, size in enumerate(sizes):
            f.write(f"Data Size: {size} rows\n")
            f.write(f"  In-Memory Peak Memory: {inmem_mem[i]:.2f}MB\n")
            f.write(f"  DuckDB Peak Memory: {duckdb_mem[i]:.2f}MB\n")
            f.write(f"  Memory Ratio (DuckDB/In-Memory): {ratios[i]:.2f}x\n\n")
            
        # Overall conclusion
        avg_ratio = sum(ratios) / len(ratios)
        f.write(f"Average Memory Ratio: {avg_ratio:.2f}x\n")
        
        if avg_ratio > 1.0:
            f.write("On average, DuckDB uses more memory than in-memory implementation\n")
        else:
            f.write("On average, DuckDB uses less memory than in-memory implementation\n")
            
        # Trend analysis
        if len(ratios) > 1:
            first_ratio = ratios[0]
            last_ratio = ratios[-1]
            
            if last_ratio < first_ratio:
                f.write("\nTrend: DuckDB's relative memory usage decreases with larger data sizes\n")
                if first_ratio > 1.0 and last_ratio < 1.0:
                    f.write("A crossover point exists where DuckDB becomes more memory-efficient\n")
            elif last_ratio > first_ratio:
                f.write("\nTrend: DuckDB's relative memory usage increases with larger data sizes\n")
                if first_ratio < 1.0 and last_ratio > 1.0:
                    f.write("A crossover point exists where in-memory becomes more memory-efficient\n")
            else:
                f.write("\nTrend: DuckDB's relative memory usage remains stable with data size\n")
    
    print(f"\nMemory test results saved to {output_dir}")
    
    # Return results
    return results_df

if __name__ == '__main__':
    # Run memory tests with a range of data sizes
    run_memory_tests(data_sizes=[1000, 2500, 5000, 7500, 10000, 15000])
    
    print("\n=== Memory Tests Complete ===")
