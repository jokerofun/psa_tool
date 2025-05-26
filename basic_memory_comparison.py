#!/usr/bin/env python3
"""
Memory Comparison: Isolated testing of memory usage between implementations.
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

def run_inmemory_test(data_size):
    """Test in-memory implementation with controlled memory measurement"""
    print(f"Testing In-Memory with {data_size} rows...")
    
    # Force garbage collection
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create test data
    df = pd.DataFrame({
        'id': range(data_size),
        'value': np.random.rand(data_size),
        'category': np.random.choice(['A', 'B', 'C'], size=data_size)
    })
    
    def process_func(dfs, params):
        # Process the data
        filtered = df[df['category'] == 'A'].copy()
        filtered['value'] = filtered['value'] * params.get('factor', 2)
        return {'result': filtered}
    
    # Setup in-memory dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    node = dataflow.node("TestNode", ParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    start_time = time.time()
    dataflow.execute({'factor': 2.0})
    execution_time = time.time() - start_time
    
    # Get peak memory
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    print(f"  Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
    
    return {
        'execution_time': execution_time,
        'peak_memory': peak_memory
    }

def run_duckdb_test(data_size):
    """Test DuckDB implementation with controlled memory measurement"""
    print(f"Testing DuckDB with {data_size} rows...")
    
    # Force garbage collection
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create test data
    df = pd.DataFrame({
        'id': range(data_size),
        'value': np.random.rand(data_size),
        'category': np.random.choice(['A', 'B', 'C'], size=data_size)
    })
    
    def process_func(dfs, params):
        # Process the data
        filtered = df[df['category'] == 'A'].copy()
        filtered['value'] = filtered['value'] * params.get('factor', 2)
        return {'result': filtered}
    
    # Setup DuckDB dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    base_dataflow = manager.newDataFlow(DemoNode)
    dataflow = DuckDBParallelDataflow(base_dataflow)
    node = base_dataflow.node("TestNode", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    start_time = time.time()
    dataflow.execute({'factor': 2.0})
    execution_time = time.time() - start_time
    
    # Get peak memory
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    
    # Stop tracking
    tracemalloc.stop()
    
    # Cleanup
    dataflow.clear_execution_data()
    manager.shutdown()
    ParallelExecutionDataFlowManager._instance = None
    
    print(f"  Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
    
    return {
        'execution_time': execution_time,
        'peak_memory': peak_memory
    }

def run_basic_comparison(data_sizes=None):
    """Run basic memory comparison tests"""
    if data_sizes is None:
        data_sizes = [1000, 5000, 10000]
    
    results = []
    
    for size in data_sizes:
        print(f"\n=== Testing with {size} rows ===")
        
        # Test in-memory first
        inmem_result = run_inmemory_test(size)
        
        # Test DuckDB next
        duckdb_result = run_duckdb_test(size)
        
        # Calculate ratio
        time_ratio = duckdb_result['execution_time'] / inmem_result['execution_time'] if inmem_result['execution_time'] > 0 else 0
        memory_ratio = duckdb_result['peak_memory'] / inmem_result['peak_memory'] if inmem_result['peak_memory'] > 0 else 0
        
        results.append({
            'data_size': size,
            'inmem_time': inmem_result['execution_time'],
            'inmem_memory': inmem_result['peak_memory'],
            'duckdb_time': duckdb_result['execution_time'],
            'duckdb_memory': duckdb_result['peak_memory'],
            'time_ratio': time_ratio,
            'memory_ratio': memory_ratio
        })
        
        print(f"Memory ratio (DuckDB/In-Memory): {memory_ratio:.2f}x")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'benchmark_results/basic_memory'
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(results).to_csv(os.path.join(output_dir, f'comparison_{timestamp}.csv'), index=False)
    
    # Print summary
    print("\n=== Summary ===")
    for result in results:
        print(f"Data Size: {result['data_size']} rows")
        print(f"  In-Memory: {result['inmem_memory']:.2f}MB")
        print(f"  DuckDB: {result['duckdb_memory']:.2f}MB")
        print(f"  Memory Ratio: {result['memory_ratio']:.2f}x\n")
    
    return results

if __name__ == "__main__":
    # Run with default sizes or from command line
    sizes = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    run_basic_comparison(sizes)
