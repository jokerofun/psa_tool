#!/usr/bin/env python3
"""
Comprehensive Dataflow Benchmark: 
Compares in-memory vs. DuckDB dataflow implementations with detailed analysis

This script provides more detailed benchmarking, analysis, and visualization
of the performance differences between in-memory and DuckDB dataflow implementations.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import threading
import psutil
import matplotlib.pyplot as plt
import traceback
import gc
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node
from src.dataflow.benchmark_utils import DataflowBenchmark

class ComprehensiveBenchmark:
    """
    Comprehensive benchmark runner with advanced metrics and visualizations
    """
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'benchmark_results',
            f'comprehensive_{self.timestamp}'
        )
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_results = []
        
    def run_benchmark_suite(self, data_sizes=None, repeat=3, log=True):
        """Run the complete benchmark suite with all data sizes"""
        if data_sizes is None:
            data_sizes = [1000, 10000, 50000, 100000]

        try:
            for size in data_sizes:
                benchmark_name = f"Dataflow_Benchmark_Size_{size}"
                print(f"\n=== Running benchmark with data size: {size} ===")
                
                # Create benchmark
                benchmark = DataflowBenchmark(benchmark_name, self.results_dir)
                
                # Run in-memory benchmark
                in_memory_result = benchmark.run_benchmark(
                    name=f"In-Memory_{size}",
                    setup_func=self._setup_inmemory_dataflow(size),
                    execute_func=self._execute_inmemory,
                    cleanup_func=self._cleanup_inmemory,
                    repeat=repeat
                )
                
                # Force garbage collection to ensure clean state
                gc.collect()
                
                # Run DuckDB benchmark
                duckdb_result = benchmark.run_benchmark(
                    name=f"DuckDB_{size}",
                    setup_func=self._setup_duckdb_dataflow(size),
                    execute_func=self._execute_duckdb,
                    cleanup_func=self._cleanup_duckdb,
                    repeat=repeat
                )
                
                # Force garbage collection
                gc.collect()
                
                # Generate and display comparison
                comparison = benchmark.compare_results()
                print("\nComparison Results:")
                print(comparison)
                
                # Generate plots
                benchmark.generate_plots()
                
                # Save results
                result_path = benchmark.save_results()
                
                # Add to all results
                self.all_results.append({
                    'data_size': size,
                    'in_memory': in_memory_result,
                    'duckdb': duckdb_result,
                    'comparison': comparison
                })
                
                # Log the results if requested
                if log:
                    self._log_benchmark_results(size, in_memory_result, duckdb_result)
                
        except Exception as e:
            print(f"Error running benchmark suite: {e}")
            traceback.print_exc()
        finally:
            # Final cleanup
            manager = ParallelExecutionDataFlowManager.getInstance()
            manager.shutdown()
            
        # Generate overall summary
        self._generate_summary()
        
    def _setup_inmemory_dataflow(self, data_size):
        """Setup function for in-memory dataflow benchmark."""
        
        def setup():
            # Get processing functions with specified data size
            funcs = self._create_processing_functions(data_size)
            
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

    def _setup_duckdb_dataflow(self, data_size):
        """Setup function for DuckDB dataflow benchmark."""
        
        def setup():
            # Get processing functions with specified data size
            funcs = self._create_processing_functions(data_size)
            
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
    
    def _execute_inmemory(self, dataflow, params):
        """Execute function for in-memory dataflow benchmark."""
        return dataflow.execute(params)

    def _execute_duckdb(self, dataflow, params):
        """Execute function for DuckDB dataflow benchmark."""
        return dataflow.execute(params)

    def _cleanup_inmemory(self, dataflow):
        """Cleanup function for in-memory dataflow benchmark."""
        # Clear node results
        for node_name, node in dataflow.nodes.items():
            if hasattr(node, 'clear_results'):
                node.clear_results()

    def _cleanup_duckdb(self, dataflow):
        """Cleanup function for DuckDB dataflow benchmark."""
        # Clear the execution data from DuckDB
        dataflow.clear_execution_data()

    def _create_processing_functions(self, data_size):
        """Create processing functions with configurable data sizes for benchmarking."""
        
        def processing_a(dfs, parameters):
            """Processing function that generates a large dataframe."""
            print(f"[{threading.current_thread().name}] Processing A: Generating {data_size} rows")
            
            # Create timestamps in chunks to avoid overflow with large data sizes
            chunk_size = 10000
            timestamps = []
            for chunk in range(0, data_size, chunk_size):
                chunk_end = min(chunk + chunk_size, data_size)
                timestamps.extend([
                    pd.Timestamp('2023-01-01') + pd.Timedelta(days=i % 365) 
                    for i in range(chunk, chunk_end)
                ])
            
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
            
        return {
            'processing_a': processing_a,
            'processing_b': processing_b,
            'processing_c': processing_c
        }

    def _log_benchmark_results(self, data_size, in_memory_result, duckdb_result):
        """Log detailed benchmark results to a file"""
        log_file = os.path.join(self.results_dir, f"benchmark_log_size_{data_size}.txt")
        
        with open(log_file, 'w') as f:
            f.write(f"=== Benchmark Results for Data Size: {data_size} ===\n\n")
            
            f.write("--- In-Memory Implementation ---\n")
            f.write(f"Average Execution Time: {in_memory_result['avg_execution_time']:.4f}s\n")
            f.write(f"Average Memory Usage: {in_memory_result['avg_memory_usage']:.2f}MB\n")
            f.write("Individual Execution Times:\n")
            for i, time_val in enumerate(in_memory_result['all_execution_times']):
                f.write(f"  Run {i+1}: {time_val:.4f}s\n")
            f.write("Individual Memory Usages:\n")
            for i, mem_val in enumerate(in_memory_result['all_memory_usages']):
                f.write(f"  Run {i+1}: {mem_val:.2f}MB\n")
            
            f.write("\n--- DuckDB Implementation ---\n")
            f.write(f"Average Execution Time: {duckdb_result['avg_execution_time']:.4f}s\n")
            f.write(f"Average Memory Usage: {duckdb_result['avg_memory_usage']:.2f}MB\n")
            f.write("Individual Execution Times:\n")
            for i, time_val in enumerate(duckdb_result['all_execution_times']):
                f.write(f"  Run {i+1}: {time_val:.4f}s\n")
            f.write("Individual Memory Usages:\n")
            for i, mem_val in enumerate(duckdb_result['all_memory_usages']):
                f.write(f"  Run {i+1}: {mem_val:.2f}MB\n")
            
            time_ratio = duckdb_result['avg_execution_time'] / in_memory_result['avg_execution_time']
            mem_ratio = duckdb_result['avg_memory_usage'] / max(in_memory_result['avg_memory_usage'], 0.001) # avoid div by zero
            
            f.write("\n--- Comparison ---\n")
            f.write(f"Time Ratio (DuckDB/In-Memory): {time_ratio:.2f}x\n")
            f.write(f"Memory Ratio (DuckDB/In-Memory): {mem_ratio:.2f}x\n")
            
            f.write(f"\nDuckDB is {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'} than In-Memory\n")
            f.write(f"DuckDB uses {mem_ratio:.2f}x {'more' if mem_ratio > 1 else 'less'} memory than In-Memory\n")

    def _generate_summary(self):
        """Generate summary of all benchmark runs with visualizations"""
        # Extract results
        data_sizes = [r['data_size'] for r in self.all_results]
        in_memory_times = [r['in_memory']['avg_execution_time'] for r in self.all_results]
        duckdb_times = [r['duckdb']['avg_execution_time'] for r in self.all_results]
        in_memory_memories = [r['in_memory']['avg_memory_usage'] for r in self.all_results]
        duckdb_memories = [r['duckdb']['avg_memory_usage'] for r in self.all_results]
        
        # Create comparison DataFrame
        summary_df = pd.DataFrame({
            'data_size': data_sizes,
            'in_memory_time': in_memory_times,
            'duckdb_time': duckdb_times,
            'in_memory_memory': in_memory_memories,
            'duckdb_memory': duckdb_memories,
            'time_ratio': [d/i if i > 0 else float('nan') for d, i in zip(duckdb_times, in_memory_times)],
            'memory_ratio': [d/i if i > 0 else float('nan') for d, i in zip(duckdb_memories, in_memory_memories)]
        })
        
        # Save summary to CSV
        summary_path = os.path.join(self.results_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Generate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Execution Time plot
        ax1.plot(data_sizes, in_memory_times, 'o-', label='In-Memory')
        ax1.plot(data_sizes, duckdb_times, 's-', label='DuckDB')
        ax1.set_xlabel('Data Size (rows)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time vs Data Size')
        ax1.legend()
        ax1.grid(True)
        
        # Memory Usage plot
        ax2.plot(data_sizes, in_memory_memories, 'o-', label='In-Memory')
        ax2.plot(data_sizes, duckdb_memories, 's-', label='DuckDB')
        ax2.set_xlabel('Data Size (rows)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Data Size')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'summary_plots.png'))
        
        # Generate log-scale plots for better visualization with large differences
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Log-scale Execution Time plot
        ax1.plot(data_sizes, in_memory_times, 'o-', label='In-Memory')
        ax1.plot(data_sizes, duckdb_times, 's-', label='DuckDB')
        ax1.set_xlabel('Data Size (rows)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time vs Data Size (Log Scale)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        # Log-scale Memory Usage plot
        ax2.plot(data_sizes, in_memory_memories, 'o-', label='In-Memory')
        ax2.plot(data_sizes, duckdb_memories, 's-', label='DuckDB')
        ax2.set_xlabel('Data Size (rows)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Data Size (Log Scale)')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'summary_plots_log_scale.png'))
        
        # Generate a text summary
        with open(os.path.join(self.results_dir, 'summary.txt'), 'w') as f:
            f.write("=== Dataflow Implementation Benchmark Summary ===\n\n")
            
            f.write("--- Performance by Data Size ---\n")
            for idx, size in enumerate(data_sizes):
                f.write(f"\nData Size: {size} rows\n")
                f.write(f"  In-Memory Execution Time: {in_memory_times[idx]:.4f}s\n")
                f.write(f"  DuckDB Execution Time: {duckdb_times[idx]:.4f}s\n")
                f.write(f"  Time Ratio (DuckDB/In-Memory): {summary_df.iloc[idx]['time_ratio']:.2f}x\n")
                f.write(f"  In-Memory Memory Usage: {in_memory_memories[idx]:.2f}MB\n")
                f.write(f"  DuckDB Memory Usage: {duckdb_memories[idx]:.2f}MB\n")
                f.write(f"  Memory Ratio (DuckDB/In-Memory): {summary_df.iloc[idx]['memory_ratio']:.2f}x\n")
            
            # General conclusions
            f.write("\n--- Overall Findings ---\n")
            
            # Time trends
            time_ratios = summary_df['time_ratio'].dropna()
            if time_ratios.empty:
                f.write("Insufficient data to analyze time trends.\n")
            else:
                avg_time_ratio = time_ratios.mean()
                f.write(f"On average, DuckDB is {avg_time_ratio:.2f}x ")
                f.write("slower than" if avg_time_ratio > 1 else "faster than")
                f.write(" In-Memory for execution time.\n")
                
                # Check if the ratio changes with data size
                if len(time_ratios) > 1:
                    time_ratio_trend = time_ratios.iloc[-1] - time_ratios.iloc[0]
                    if abs(time_ratio_trend) > 0.1:  # Significant trend
                        f.write(f"The performance gap {'increases' if time_ratio_trend > 0 else 'decreases'} ")
                        f.write(f"with larger data sizes.\n")
            
            # Memory trends
            memory_ratios = summary_df['memory_ratio'].dropna()
            if memory_ratios.empty:
                f.write("Insufficient data to analyze memory trends.\n")
            else:
                avg_memory_ratio = memory_ratios.mean()
                f.write(f"On average, DuckDB uses {avg_memory_ratio:.2f}x ")
                f.write("more memory than" if avg_memory_ratio > 1 else "less memory than")
                f.write(" In-Memory.\n")
                
                # Check if the ratio changes with data size
                if len(memory_ratios) > 1:
                    memory_ratio_trend = memory_ratios.iloc[-1] - memory_ratios.iloc[0]
                    if abs(memory_ratio_trend) > 0.1:  # Significant trend
                        f.write(f"The memory usage difference {'increases' if memory_ratio_trend > 0 else 'decreases'} ")
                        f.write(f"with larger data sizes.\n")
            
            # Recommendations
            f.write("\n--- Recommendations ---\n")
            if not time_ratios.empty and not memory_ratios.empty:
                if avg_time_ratio <= 1.2 and avg_memory_ratio <= 1.2:
                    f.write("Both implementations perform similarly. Choose based on other factors like persistence needs.\n")
                elif avg_time_ratio <= 1.2 and avg_memory_ratio > 1.2:
                    f.write("In-Memory has memory advantages with similar execution time. Prefer In-Memory unless persistence is needed.\n")
                elif avg_time_ratio > 1.2 and avg_memory_ratio <= 1.2:
                    f.write("In-Memory has speed advantages with similar memory usage. Prefer In-Memory unless persistence is needed.\n")
                else:
                    f.write("In-Memory performs better in both time and memory. Only use DuckDB when persistence is required.\n")
                    
                # Check for crossover points
                if len(data_sizes) > 1:
                    f.write("\nConsider the specific data size for your use case:\n")
                    for i in range(len(data_sizes)-1):
                        # Check if there's a crossover in performance between these data sizes
                        time1, time2 = time_ratios.iloc[i], time_ratios.iloc[i+1]
                        mem1, mem2 = memory_ratios.iloc[i], memory_ratios.iloc[i+1]
                        
                        if (time1 < 1 and time2 > 1) or (time1 > 1 and time2 < 1):
                            f.write(f"  Time performance crossover occurs between {data_sizes[i]} and {data_sizes[i+1]} rows.\n")
                        if (mem1 < 1 and mem2 > 1) or (mem1 > 1 and mem2 < 1):
                            f.write(f"  Memory usage crossover occurs between {data_sizes[i]} and {data_sizes[i+1]} rows.\n")
        
        print(f"\nBenchmark summary saved to: {self.results_dir}")
        return summary_df

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

if __name__ == "__main__":
    # Define data sizes to test
    data_sizes = [1000, 5000, 10000, 25000, 50000]
    
    print("Starting comprehensive benchmark...")
    benchmark = ComprehensiveBenchmark()
    benchmark.run_benchmark_suite(data_sizes=data_sizes, repeat=3)
    
    print("\n=== Comprehensive Benchmark Complete ===")
    print(f"Results saved in {benchmark.results_dir}")
