#!/usr/bin/env python3
"""
Direct Memory Comparison: Simple and direct comparison of memory usage
between in-memory and DuckDB implementations with specific data sizes.
"""

import pandas as pd
import numpy as np
import tracemalloc
import gc
import os
import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node

# Simple Node class for testing
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

def create_dataframe(size):
    """Create a test dataframe of specified size"""
    data = {
        'id': range(size),
        'value': np.random.rand(size),
        'category': np.random.choice(['A', 'B', 'C'], size=size)
    }
    return pd.DataFrame(data)

def measure_memory_usage():
    """Measure current memory usage in MB"""
    return tracemalloc.get_traced_memory()[1] / (1024 * 1024)

def test_in_memory_dataflow(data_size):
    """Test memory usage of in-memory dataflow"""
    print(f"Testing in-memory dataflow with {data_size} rows...")
    
    # Create test dataframe
    df = create_dataframe(data_size)
    
    # Define processing function
    def process_func(dfs, params):
        result = df[df['category'] == 'A'].copy()
        result['value'] = result['value'] * 2
        return {'result': result}
    
    # Start memory tracking
    tracemalloc.start()
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    
    # Create node
    node = dataflow.node("TestNode", ParallelExecutionNode)
    node.process_func = process_func
    
    # Execute and measure memory
    start_time = time.time()
    _ = dataflow.execute({})
    end_time = time.time()
    
    # Measure peak memory
    peak_memory = measure_memory_usage()
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    execution_time = end_time - start_time
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Peak memory: {peak_memory:.2f}MB")
    
    return {
        'execution_time': execution_time,
        'peak_memory': peak_memory
    }

def test_duckdb_dataflow(data_size):
    """Test memory usage of DuckDB dataflow"""
    print(f"Testing DuckDB dataflow with {data_size} rows...")
    
    # Create test dataframe
    df = create_dataframe(data_size)
    
    # Define processing function
    def process_func(dfs, params):
        result = df[df['category'] == 'A'].copy()
        result['value'] = result['value'] * 2
        return {'result': result}
    
    # Start memory tracking
    tracemalloc.start()
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    duckdb_dataflow = DuckDBParallelDataflow(dataflow)
    
    # Create node
    node = dataflow.node("TestNode", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Execute and measure memory
    start_time = time.time()
    _ = duckdb_dataflow.execute({})
    end_time = time.time()
    
    # Measure peak memory
    peak_memory = measure_memory_usage()
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    duckdb_dataflow.clear_execution_data()
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    execution_time = end_time - start_time
    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Peak memory: {peak_memory:.2f}MB")
    
    return {
        'execution_time': execution_time,
        'peak_memory': peak_memory
    }

def run_direct_comparison(sizes=None):
    """Run direct comparison between implementations"""
    if sizes is None:
        sizes = [1000, 5000, 10000]
    
    results = []
    
    for size in sizes:
        print(f"\n=== Testing with data size: {size} ===")
        
        # Force garbage collection
        gc.collect()
        
        # Test in-memory
        inmem_result = test_in_memory_dataflow(size)
        
        # Force garbage collection
        gc.collect()
        
        # Test DuckDB
        duckdb_result = test_duckdb_dataflow(size)
        
        # Calculate ratios
        time_ratio = duckdb_result['execution_time'] / inmem_result['execution_time']
        memory_ratio = duckdb_result['peak_memory'] / inmem_result['peak_memory']
        
        print("\nComparison:")
        print(f"  Time ratio (DuckDB/In-Memory): {time_ratio:.2f}x")
        print(f"  Memory ratio (DuckDB/In-Memory): {memory_ratio:.2f}x")
        
        # Save result
        results.append({
            'data_size': size,
            'in_memory': inmem_result,
            'duckdb': duckdb_result,
            'time_ratio': time_ratio,
            'memory_ratio': memory_ratio
        })
    
    # Save results to CSV
    output_dir = 'benchmark_results/direct_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame
    results_df = pd.DataFrame([
        {
            'data_size': r['data_size'],
            'inmem_time': r['in_memory']['execution_time'],
            'duckdb_time': r['duckdb']['execution_time'],
            'inmem_memory': r['in_memory']['peak_memory'],
            'duckdb_memory': r['duckdb']['peak_memory'],
            'time_ratio': r['time_ratio'],
            'memory_ratio': r['memory_ratio']
        } for r in results
    ])
    
    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, f'comparison_{timestamp}.csv'), index=False)
    
    # Create summary text
    with open(os.path.join(output_dir, f'summary_{timestamp}.txt'), 'w') as f:
        f.write("=== Direct Memory Comparison Results ===\n\n")
        
        for result in results:
            f.write(f"Data Size: {result['data_size']} rows\n")
            f.write(f"  In-Memory:\n")
            f.write(f"    Execution Time: {result['in_memory']['execution_time']:.4f}s\n")
            f.write(f"    Peak Memory: {result['in_memory']['peak_memory']:.2f}MB\n")
            f.write(f"  DuckDB:\n")
            f.write(f"    Execution Time: {result['duckdb']['execution_time']:.4f}s\n")
            f.write(f"    Peak Memory: {result['duckdb']['peak_memory']:.2f}MB\n")
            f.write(f"  Ratios:\n")
            f.write(f"    Time Ratio (DuckDB/In-Memory): {result['time_ratio']:.2f}x\n")
            f.write(f"    Memory Ratio (DuckDB/In-Memory): {result['memory_ratio']:.2f}x\n\n")
    
    print(f"\nResults saved to {output_dir}")
    
    return results

if __name__ == '__main__':
    run_direct_comparison(sizes=[1000, 2500, 5000, 7500, 10000])
