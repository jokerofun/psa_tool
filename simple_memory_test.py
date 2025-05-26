#!/usr/bin/env python3
"""
Simple Memory Test: A very basic script to test memory usage without complex dependencies.
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

def measure_memory():
    """Measure current memory usage in MB"""
    return tracemalloc.get_traced_memory()[1] / (1024 * 1024)

def create_test_dataframe(size):
    """Create a simple test dataframe"""
    print(f"Creating test dataframe with {size} rows")
    data = {
        'id': range(size),
        'value': np.random.rand(size),
        'category': np.random.choice(['A', 'B', 'C'], size=size)
    }
    return pd.DataFrame(data)

def test_duckdb_memory(data_size):
    """Test DuckDB implementation memory usage"""
    print(f"\nTesting DuckDB with {data_size} rows...")
    
    # Force garbage collection
    gc.collect()
    
    # Create test data outside of measurement
    df = create_test_dataframe(data_size)
    
    def process_func(dfs, params):
        result = df[df['category'] == 'A'].copy()
        return {'result': result}
    
    # Start memory tracking
    tracemalloc.start()
    
    # Setup DuckDB dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    base_dataflow = manager.newDataFlow(DemoNode)
    dataflow = DuckDBParallelDataflow(base_dataflow)
    
    # Create a simple node
    node = base_dataflow.node("Test", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    dataflow.execute({})
    
    # Get memory usage
    peak_memory = measure_memory()
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    dataflow.clear_execution_data()
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    print(f"  Peak memory: {peak_memory:.2f} MB")
    return peak_memory

def test_inmemory_memory(data_size):
    """Test in-memory implementation memory usage"""
    print(f"\nTesting In-Memory with {data_size} rows...")
    
    # Force garbage collection
    gc.collect()
    
    # Create test data outside of measurement
    df = create_test_dataframe(data_size)
    
    def process_func(dfs, params):
        result = df[df['category'] == 'A'].copy()
        return {'result': result}
    
    # Start memory tracking
    tracemalloc.start()
    
    # Setup in-memory dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    
    # Create a simple node
    node = dataflow.node("Test", ParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    dataflow.execute({})
    
    # Get memory usage
    peak_memory = measure_memory()
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    print(f"  Peak memory: {peak_memory:.2f} MB")
    return peak_memory

def run_tests():
    """Run memory tests for different data sizes"""
    data_sizes = [1000, 5000, 10000]
    results = []
    
    for size in data_sizes:
        print(f"\n=== Testing with data size: {size} ===")
        
        # Test in-memory first
        inmem_memory = test_inmemory_memory(size)
        
        # Force garbage collection
        gc.collect()
        
        # Test DuckDB
        duckdb_memory = test_duckdb_memory(size)
        
        # Calculate ratio
        ratio = duckdb_memory / inmem_memory if inmem_memory > 0 else 0
        
        results.append({
            'data_size': size,
            'inmem_memory': inmem_memory,
            'duckdb_memory': duckdb_memory,
            'ratio': ratio
        })
        
        print(f"\nMemory usage for {size} rows:")
        print(f"  In-Memory: {inmem_memory:.2f} MB")
        print(f"  DuckDB: {duckdb_memory:.2f} MB")
        print(f"  Ratio (DuckDB/In-Memory): {ratio:.2f}x")
    
    # Print final summary
    print("\n=== Memory Test Summary ===")
    for result in results:
        print(f"Data Size: {result['data_size']} rows")
        print(f"  In-Memory: {result['inmem_memory']:.2f} MB")
        print(f"  DuckDB: {result['duckdb_memory']:.2f} MB")
        print(f"  Ratio: {result['ratio']:.2f}x\n")

if __name__ == '__main__':
    # Run tests
    run_tests()
