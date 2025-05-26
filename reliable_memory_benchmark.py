#!/usr/bin/env python3
"""
Reliable Memory Benchmark: A robust script for accurately comparing memory usage
between in-memory and DuckDB dataflow implementations.

This script focuses on obtaining consistent and reliable memory measurements
by isolating each test run, properly controlling garbage collection, and using
consistent memory measurement techniques.
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
import psutil
import subprocess
import json
from statistics import mean, stdev

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

class ReliableMemoryBenchmark:
    """
    A benchmark class that provides reliable and consistent memory measurements
    for comparing in-memory and DuckDB implementations.
    """
    
    def __init__(self, output_dir='benchmark_results/reliable'):
        """Initialize benchmark with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def create_test_data(self, size):
        """Create test dataframe with specified number of rows"""
        print(f"Creating test dataframe with {size} rows")
        
        # Create in chunks to avoid memory issues with large datasets
        chunk_size = 10000
        chunks = []
        
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            chunk_length = end - i
            
            chunk = pd.DataFrame({
                'id': range(i, end),
                'timestamp': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=j % 365) for j in range(i, end)],
                'value': np.random.rand(chunk_length),
                'category': np.random.choice(['A', 'B', 'C'], size=chunk_length)
            })
            chunks.append(chunk)
        
        # Combine chunks
        if len(chunks) == 1:
            return chunks[0]
        else:
            return pd.concat(chunks)
    
    def _run_in_subprocess(self, test_type, data_size):
        """
        Run a specific test in a separate process to ensure memory isolation.
        
        Args:
            test_type: Type of test to run ('inmemory' or 'duckdb')
            data_size: Size of the test data
            
        Returns:
            dict: Test results including execution time and memory usage
        """
        script_content = f"""
import sys
import os
import gc
import time
import pandas as pd
import numpy as np
import tracemalloc
import json

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

def create_test_data(size):
    """Create test dataframe with specified number of rows"""
    # Create in chunks to avoid memory issues with large datasets
    chunk_size = 10000
    chunks = []
    
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        chunk_length = end - i
        
        chunk = pd.DataFrame({
            'id': range(i, end),
            'timestamp': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=j % 365) for j in range(i, end)],
            'value': np.random.rand(chunk_length),
            'category': np.random.choice(['A', 'B', 'C'], size=chunk_length)
        })
        chunks.append(chunk)
    
    # Combine chunks
    if len(chunks) == 1:
        return chunks[0]
    else:
        return pd.concat(chunks)

def measure_memory_usage():
    """Get current memory usage in MB"""
    return tracemalloc.get_traced_memory()[1] / (1024 * 1024)

# Force garbage collection at start
gc.collect()

# Create test data
data_size = {data_size}
test_df = create_test_data(data_size)

# Define processing
def process_func(dfs, params):
    result = test_df[test_df['category'] == 'A'].copy()
    result['value'] = result['value'] * params.get('factor', 2)
    return {{'result': result}}

# Start memory tracking
tracemalloc.start()

# Start timing
start_time = time.time()

if "{test_type}" == "inmemory":
    # Setup in-memory dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(DemoNode)
    
    # Create a simple node
    node = dataflow.node("Test", ParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    dataflow.execute({{'factor': 2.0}})
    
else:  # duckdb
    # Setup DuckDB dataflow
    manager = ParallelExecutionDataFlowManager.getInstance()
    base_dataflow = manager.newDataFlow(DemoNode)
    dataflow = DuckDBParallelDataflow(base_dataflow)
    
    # Create a simple node
    node = base_dataflow.node("Test", DuckDBParallelExecutionNode)
    node.process_func = process_func
    
    # Execute
    dataflow.execute({{'factor': 2.0}})
    
    # Clear execution data
    dataflow.clear_execution_data()

# End timing
end_time = time.time()

# Get memory usage
peak_memory = measure_memory_usage()

# Stop tracking
tracemalloc.stop()

# Print results as JSON
result = {{
    'execution_time': end_time - start_time,
    'peak_memory': peak_memory
}}

print(json.dumps(result))
        """
        
        # Create a temporary script file
        temp_script = f"temp_{test_type}_{data_size}.py"
        with open(temp_script, "w") as f:
            f.write(script_content)
        
        try:
            # Run the script in a separate process
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True
            )
            
            # Parse the output
            output = result.stdout.strip()
            result_data = json.loads(output)
            return result_data
            
        except Exception as e:
            print(f"Error running subprocess: {e}")
            print(f"Stderr: {result.stderr if 'result' in locals() else 'No stderr'}")
            return {'execution_time': 0, 'peak_memory': 0}
        finally:
            # Clean up the temporary script
            if os.path.exists(temp_script):
                os.remove(temp_script)
    
    def run_benchmark(self, data_size, repeat=5):
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
            inmem_result = self._run_in_subprocess('inmemory', data_size)
            
            execution_time = inmem_result['execution_time']
            peak_memory = inmem_result['peak_memory']
            
            inmem_times.append(execution_time)
            inmem_memories.append(peak_memory)
            
            print(f"      Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
            
            # Wait a bit between tests to ensure process cleanup
            time.sleep(1)
            
            # ===== DuckDB Implementation =====
            print("    Testing DuckDB implementation...")
            duckdb_result = self._run_in_subprocess('duckdb', data_size)
            
            execution_time = duckdb_result['execution_time']
            peak_memory = duckdb_result['peak_memory']
            
            duckdb_times.append(execution_time)
            duckdb_memories.append(peak_memory)
            
            print(f"      Time: {execution_time:.4f}s, Memory: {peak_memory:.2f}MB")
            
            # Wait a bit between tests to ensure process cleanup
            time.sleep(1)
        
        # Calculate averages
        avg_inmem_time = mean(inmem_times)
        avg_inmem_memory = mean(inmem_memories)
        avg_duckdb_time = mean(duckdb_times)
        avg_duckdb_memory = mean(duckdb_memories)
        
        # Calculate standard deviations
        std_inmem_time = stdev(inmem_times) if len(inmem_times) > 1 else 0
        std_inmem_memory = stdev(inmem_memories) if len(inmem_memories) > 1 else 0
        std_duckdb_time = stdev(duckdb_times) if len(duckdb_times) > 1 else 0
        std_duckdb_memory = stdev(duckdb_memories) if len(duckdb_memories) > 1 else 0
        
        # Calculate ratios
        time_ratio = avg_duckdb_time / avg_inmem_time if avg_inmem_time > 0 else 0
        memory_ratio = avg_duckdb_memory / avg_inmem_memory if avg_inmem_memory > 0 else 0
        
        # Create result record
        result = {
            'data_size': data_size,
            'in_memory': {
                'avg_execution_time': avg_inmem_time,
                'std_execution_time': std_inmem_time,
                'avg_memory_usage': avg_inmem_memory,
                'std_memory_usage': std_inmem_memory,
                'all_execution_times': inmem_times,
                'all_memory_usages': inmem_memories
            },
            'duckdb': {
                'avg_execution_time': avg_duckdb_time,
                'std_execution_time': std_duckdb_time,
                'avg_memory_usage': avg_duckdb_memory,
                'std_memory_usage': std_duckdb_memory,
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
        print(f"    In-Memory: Time={avg_inmem_time:.4f}s (±{std_inmem_time:.4f}), Memory={avg_inmem_memory:.2f}MB (±{std_inmem_memory:.2f})")
        print(f"    DuckDB: Time={avg_duckdb_time:.4f}s (±{std_duckdb_time:.4f}), Memory={avg_duckdb_memory:.2f}MB (±{std_duckdb_memory:.2f})")
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
            data_sizes = [1000, 2500, 5000, 10000, 15000]
        
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
        
        # Error bars data
        inmem_time_err = [r['in_memory']['std_execution_time'] for r in self.results]
        duckdb_time_err = [r['duckdb']['std_execution_time'] for r in self.results]
        inmem_mem_err = [r['in_memory']['std_memory_usage'] for r in self.results]
        duckdb_mem_err = [r['duckdb']['std_memory_usage'] for r in self.results]
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV data
        results_df = pd.DataFrame({
            'data_size': data_sizes,
            'inmem_time': inmem_times,
            'inmem_time_std': inmem_time_err,
            'duckdb_time': duckdb_times,
            'duckdb_time_std': duckdb_time_err,
            'time_ratio': time_ratios,
            'inmem_memory': inmem_memories,
            'inmem_memory_std': inmem_mem_err,
            'duckdb_memory': duckdb_memories,
            'duckdb_memory_std': duckdb_mem_err,
            'memory_ratio': memory_ratios
        })
        
        results_df.to_csv(os.path.join(self.output_dir, f'benchmark_results_{timestamp}.csv'), index=False)
        
        # Create execution time plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(data_sizes, inmem_times, yerr=inmem_time_err, fmt='o-', capsize=5, label='In-Memory')
        plt.errorbar(data_sizes, duckdb_times, yerr=duckdb_time_err, fmt='s-', capsize=5, label='DuckDB')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'time_comparison_{timestamp}.png'))
        
        # Create memory usage plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(data_sizes, inmem_memories, yerr=inmem_mem_err, fmt='o-', capsize=5, label='In-Memory')
        plt.errorbar(data_sizes, duckdb_memories, yerr=duckdb_mem_err, fmt='s-', capsize=5, label='DuckDB')
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
            f.write("=== Reliable Memory Benchmark Results ===\n\n")
            
            for result in self.results:
                size = result['data_size']
                f.write(f"Data Size: {size} rows\n")
                f.write(f"  In-Memory:\n")
                f.write(f"    Execution Time: {result['in_memory']['avg_execution_time']:.4f}s ± {result['in_memory']['std_execution_time']:.4f}\n")
                f.write(f"    Memory Usage: {result['in_memory']['avg_memory_usage']:.2f}MB ± {result['in_memory']['std_memory_usage']:.2f}\n")
                f.write(f"  DuckDB:\n")
                f.write(f"    Execution Time: {result['duckdb']['avg_execution_time']:.4f}s ± {result['duckdb']['std_execution_time']:.4f}\n")
                f.write(f"    Memory Usage: {result['duckdb']['avg_memory_usage']:.2f}MB ± {result['duckdb']['std_memory_usage']:.2f}\n")
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


if __name__ == "__main__":
    # Create benchmark instance
    benchmark = ReliableMemoryBenchmark()
    
    # Run benchmarks with multiple data sizes
    benchmark.run_multiple_sizes(data_sizes=[1000, 5000, 10000], repeat=3)
    
    print("\n=== Memory Benchmark Complete ===")
