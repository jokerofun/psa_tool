#!/usr/bin/env python3
"""
Improved Benchmark: Measures memory usage more accurately using proper memory isolation and
consistent measurement techniques, with garbage collection controls.
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
import gc
import tracemalloc

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

class ImprovedBenchmark:
    """
    An improved benchmarking class that provides more accurate memory measurements
    and more controlled execution environment.
    """
    
    def __init__(self, output_dir='benchmark_results/improved'):
        """Initialize benchmark with name and output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def measure_peak_memory(self, func, *args, **kwargs):
        """
        Measure peak memory usage of a function with tracemalloc for more accurate results
        
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
    
    def run_benchmark(self, data_size, repeat=3):
        """
        Run benchmark for both implementations with specified data size
        
        Args:
            data_size: Number of rows in test data
            repeat: Number of repetitions to run
        
        Returns:
            dict: Benchmark results
        """
        print(f"\n=== Running benchmark with data size: {data_size} ===")
        
        inmem_times = []
        inmem_memories = []
        duckdb_times = []
        duckdb_memories = []
        
        for i in range(repeat):
            print(f"  Run {i+1}/{repeat}...")
            
            # ===== In-Memory Implementation =====
            print("    Testing In-Memory implementation...")
            
            # Create processing functions
            funcs = self._create_processing_functions(data_size)
            
            # Setup in-memory dataflow
            def setup_inmem():
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
                
                return dataflow, {'factor_a': 2.0, 'factor_b': 3.0}
            
            # Execute in-memory with memory measurement
            dataflow, params = setup_inmem()
            
            start_time = time.time()
            _, peak_memory = self.measure_peak_memory(
                lambda: dataflow.execute(params)
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            inmem_times.append(execution_time)
            inmem_memories.append(peak_memory)
            
            print(f"      Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
            
            # Cleanup
            ParallelExecutionDataFlowManager.getInstance().shutdown()
            ParallelExecutionDataFlowManager._instance = None
            gc.collect()
            
            # ===== DuckDB Implementation =====
            print("    Testing DuckDB implementation...")
            
            # Setup DuckDB dataflow
            def setup_duckdb():
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
                
                return duckdb_dataflow, {'factor_a': 2.0, 'factor_b': 3.0}
            
            # Execute DuckDB with memory measurement
            dataflow, params = setup_duckdb()
            
            start_time = time.time()
            _, peak_memory = self.measure_peak_memory(
                lambda: dataflow.execute(params)
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Clear execution data
            dataflow.clear_execution_data()
            
            duckdb_times.append(execution_time)
            duckdb_memories.append(peak_memory)
            
            print(f"      Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
            
            # Cleanup
            ParallelExecutionDataFlowManager.getInstance().shutdown()
            ParallelExecutionDataFlowManager._instance = None
            gc.collect()
        
        # Calculate averages
        avg_inmem_time = sum(inmem_times) / len(inmem_times)
        avg_inmem_memory = sum(inmem_memories) / len(inmem_memories)
        avg_duckdb_time = sum(duckdb_times) / len(duckdb_times)
        avg_duckdb_memory = sum(duckdb_memories) / len(duckdb_memories)
        
        # Calculate ratios
        time_ratio = avg_duckdb_time / avg_inmem_time
        memory_ratio = avg_duckdb_memory / avg_inmem_memory
        
        # Create result record
        result = {
            'data_size': data_size,
            'in_memory': {
                'avg_execution_time': avg_inmem_time,
                'avg_memory_usage': avg_inmem_memory,
                'all_execution_times': inmem_times,
                'all_memory_usages': inmem_memories
            },
            'duckdb': {
                'avg_execution_time': avg_duckdb_time,
                'avg_memory_usage': avg_duckdb_memory,
                'all_execution_times': duckdb_times,
                'all_memory_usages': duckdb_memories
            },
            'ratios': {
                'time_ratio': time_ratio,
                'memory_ratio': memory_ratio
            }
        }
        
        self.results.append(result)
        
        print("\n  Average Results:")
        print(f"    In-Memory: Time={avg_inmem_time:.4f}s, Memory={avg_inmem_memory:.2f}MB")
        print(f"    DuckDB: Time={avg_duckdb_time:.4f}s, Memory={avg_duckdb_memory:.2f}MB")
        print(f"    Ratios: Time={time_ratio:.2f}x, Memory={memory_ratio:.2f}x")
        
        return result
    
    def run_multiple_sizes(self, data_sizes=None, repeat=3):
        """
        Run benchmarks with multiple data sizes
        
        Args:
            data_sizes: List of data sizes to test
            repeat: Number of repetitions for each size
        """
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000, 25000]
        
        for size in data_sizes:
            self.run_benchmark(size, repeat)
            
        self.generate_report()
    
    def generate_report(self):
        """Generate report with visualizations and summary"""
        if not self.results:
            print("No results to report")
            return
        
        # Extract data for plotting
        data_sizes = [r['data_size'] for r in self.results]
        inmem_times = [r['in_memory']['avg_execution_time'] for r in self.results]
        duckdb_times = [r['duckdb']['avg_execution_time'] for r in self.results]
        inmem_memories = [r['in_memory']['avg_memory_usage'] for r in self.results]
        duckdb_memories = [r['duckdb']['avg_memory_usage'] for r in self.results]
        time_ratios = [r['ratios']['time_ratio'] for r in self.results]
        memory_ratios = [r['ratios']['memory_ratio'] for r in self.results]
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV data
        results_df = pd.DataFrame({
            'data_size': data_sizes,
            'inmem_time': inmem_times,
            'duckdb_time': duckdb_times,
            'time_ratio': time_ratios,
            'inmem_memory': inmem_memories,
            'duckdb_memory': duckdb_memories,
            'memory_ratio': memory_ratios
        })
        
        results_df.to_csv(os.path.join(self.output_dir, f'benchmark_results_{timestamp}.csv'), index=False)
        
        # Create execution time plot
        plt.figure(figsize=(10, 6))
        plt.plot(data_sizes, inmem_times, 'o-', label='In-Memory')
        plt.plot(data_sizes, duckdb_times, 's-', label='DuckDB')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'time_comparison_{timestamp}.png'))
        
        # Create memory usage plot
        plt.figure(figsize=(10, 6))
        plt.plot(data_sizes, inmem_memories, 'o-', label='In-Memory')
        plt.plot(data_sizes, duckdb_memories, 's-', label='DuckDB')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'memory_comparison_{timestamp}.png'))
        
        # Create ratio plots
        plt.figure(figsize=(10, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(data_sizes, time_ratios, 'o-', color='red')
        plt.axhline(y=1.0, linestyle='--', color='gray')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Time Ratio (DuckDB/In-Memory)')
        plt.title('Execution Time Ratio')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(data_sizes, memory_ratios, 'o-', color='green')
        plt.axhline(y=1.0, linestyle='--', color='gray')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Memory Ratio (DuckDB/In-Memory)')
        plt.title('Memory Usage Ratio')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'ratios_{timestamp}.png'))
        
        # Generate text summary
        with open(os.path.join(self.output_dir, f'summary_{timestamp}.txt'), 'w') as f:
            f.write("=== Improved Benchmark Results ===\n\n")
            
            for result in self.results:
                size = result['data_size']
                f.write(f"Data Size: {size} rows\n")
                f.write(f"  In-Memory:\n")
                f.write(f"    Execution Time: {result['in_memory']['avg_execution_time']:.4f}s\n")
                f.write(f"    Memory Usage: {result['in_memory']['avg_memory_usage']:.2f}MB\n")
                f.write(f"  DuckDB:\n")
                f.write(f"    Execution Time: {result['duckdb']['avg_execution_time']:.4f}s\n")
                f.write(f"    Memory Usage: {result['duckdb']['avg_memory_usage']:.2f}MB\n")
                f.write(f"  Ratios:\n")
                f.write(f"    Time Ratio (DuckDB/In-Memory): {result['ratios']['time_ratio']:.2f}x\n")
                f.write(f"    Memory Ratio (DuckDB/In-Memory): {result['ratios']['memory_ratio']:.2f}x\n\n")
            
            # Overall analysis
            avg_time_ratio = sum(time_ratios) / len(time_ratios)
            avg_memory_ratio = sum(memory_ratios) / len(memory_ratios)
            
            f.write("=== Overall Analysis ===\n\n")
            f.write(f"Average Time Ratio: {avg_time_ratio:.2f}x\n")
            f.write(f"Average Memory Ratio: {avg_memory_ratio:.2f}x\n\n")
            
            # Memory trend analysis
            if len(memory_ratios) > 1:
                first_ratio = memory_ratios[0]
                last_ratio = memory_ratios[-1]
                
                if last_ratio < first_ratio:
                    f.write("Memory Trend: DuckDB's relative memory usage decreases with larger data sizes\n")
                elif last_ratio > first_ratio:
                    f.write("Memory Trend: DuckDB's relative memory usage increases with larger data sizes\n")
                else:
                    f.write("Memory Trend: DuckDB's relative memory usage remains stable with larger data sizes\n")
        
        print(f"\nReport generated in {self.output_dir}")
    
    def _create_processing_functions(self, data_size):
        """Create processing functions with configurable data sizes"""
        def processing_a(dfs, parameters):
            """Processing function that generates a large dataframe"""
            print(f"Processing A: Generating {data_size} rows")
            
            # Create test data in chunks to avoid memory issues with large data
            chunk_size = 10000
            data = {
                'timestamp': [],
                'values': [],
                'category': []
            }
            
            for i in range(0, data_size, chunk_size):
                end = min(i + chunk_size, data_size)
                chunk_length = end - i
                
                # Add timestamps
                data['timestamp'].extend([
                    pd.Timestamp('2023-01-01') + pd.Timedelta(days=(i + j) % 365)
                    for j in range(chunk_length)
                ])
                
                # Add values and categories
                data['values'].extend(np.random.rand(chunk_length) * parameters.get('factor_a', 1.0))
                data['category'].extend(np.random.choice(['A', 'B', 'C'], size=chunk_length))
            
            # Create the dataframe
            result_df = pd.DataFrame(data)
            
            print(f"Processing A complete")
            return {'results': result_df}

        def processing_b(dfs, parameters):
            """Processing function that filters data from node A"""
            print(f"Processing B: Filtering data")
            
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
            
            print(f"Processing B complete with {len(result_df)} rows")
            return {'filtered_results': result_df}

        def processing_c(dfs, parameters):
            """Processing function that aggregates data from node B"""
            print(f"Processing C: Aggregating data")
            
            input_df = None
            for key, df in dfs.items():
                input_df = df
                break
                
            if input_df is None or input_df.empty:
                print("Warning: No input data for processing C")
                return {'results': pd.DataFrame()}
            
            # Aggregate by date
            result_df = input_df.resample('D', on='timestamp').agg({'values': 'sum'}).reset_index()
            
            print(f"Processing C complete with {len(result_df)} rows")
            return {'aggregated_results': result_df}
            
        return {
            'processing_a': processing_a,
            'processing_b': processing_b,
            'processing_c': processing_c
        }

if __name__ == '__main__':
    # Run improved benchmark with multiple data sizes
    benchmark = ImprovedBenchmark()
    benchmark.run_multiple_sizes(data_sizes=[1000, 5000, 10000], repeat=3)
    
    print("\n=== Improved Benchmark Complete ===")
