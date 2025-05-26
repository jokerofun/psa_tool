#!/usr/bin/env python3
"""
Minimal Benchmark: Performs a minimal benchmark to test both implementations
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

def create_test_df(size=10000):
    """Create a test dataframe of specified size"""
    data = {
        'id': range(size),
        'value': np.random.rand(size),
        'category': np.random.choice(['A', 'B', 'C'], size=size)
    }
    return pd.DataFrame(data)

def test_inmemory():
    """Test in-memory dataflow"""
    print("\nTesting In-Memory Implementation:")
    
    # Create test dataframe
    test_df = create_test_df(10000)
    print(f"Test dataframe created with {len(test_df)} rows")
    
    # Measure memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    
    # Create a simple node
    def process_func(dfs, params):
        # Just do some work and return
        df = test_df.copy()
        result = df[df['category'] == 'A'].copy()
        result['value'] = result['value'] * 2
        print(f"Processed {len(result)} rows")
        return {'result': result}
    
    node = dataflow.node("TestNode", ParallelExecutionNode)
    node.process_func = process_func
    
    # Execute and time
    start_time = time.time()
    result = dataflow.execute({})
    end_time = time.time()
    
    # Measure memory after
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    
    print(f"Execution time: {end_time - start_time:.4f}s")
    print(f"Memory usage: {memory_after - memory_before:.2f}MB")
    
    return {
        'execution_time': end_time - start_time,
        'memory_usage': memory_after - memory_before
    }

def test_duckdb():
    """Test DuckDB dataflow"""
    print("\nTesting DuckDB Implementation:")
    
    # Create test dataframe
    test_df = create_test_df(10000)
    print(f"Test dataframe created with {len(test_df)} rows")
    
    # Measure memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Setup dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    duckdb_dataflow = DuckDBParallelDataflow(dataflow)
    
    # Create a simple node
    def process_func(dfs, params):
        # Just do some work and return
        df = test_df.copy()
        result = df[df['category'] == 'A'].copy()
        result['value'] = result['value'] * 2
        print(f"Processed {len(result)} rows")
        return {'result': result}
    
    node = dataflow.node("TestNode", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Execute and time
    start_time = time.time()
    result = duckdb_dataflow.execute({})
    end_time = time.time()
    
    # Measure memory after
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    
    print(f"Execution time: {end_time - start_time:.4f}s")
    print(f"Memory usage: {memory_after - memory_before:.2f}MB")
    
    # Clean up
    duckdb_dataflow.clear_execution_data()
    
    return {
        'execution_time': end_time - start_time,
        'memory_usage': memory_after - memory_before
    }

def run_comparison():
    """Run a simple comparison between implementations"""
    # Test in-memory
    inmemory_results = test_inmemory()
    
    # Test DuckDB
    duckdb_results = test_duckdb()
    
    # Compare results
    time_ratio = duckdb_results['execution_time'] / inmemory_results['execution_time']
    memory_ratio = duckdb_results['memory_usage'] / max(inmemory_results['memory_usage'], 0.001)
    
    print("\n--- Comparison Results ---")
    print(f"In-Memory execution time: {inmemory_results['execution_time']:.4f}s")
    print(f"DuckDB execution time: {duckdb_results['execution_time']:.4f}s")
    print(f"Time ratio (DuckDB/In-Memory): {time_ratio:.2f}x")
    
    print(f"In-Memory memory usage: {inmemory_results['memory_usage']:.2f}MB")
    print(f"DuckDB memory usage: {duckdb_results['memory_usage']:.2f}MB")
    print(f"Memory ratio (DuckDB/In-Memory): {memory_ratio:.2f}x")

if __name__ == "__main__":
    run_comparison()
    
    # Clean up
    manager = ParallelExecutionDataFlowManager.getInstance()
    manager.shutdown()
