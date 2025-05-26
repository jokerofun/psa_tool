import time
import psutil
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

__all__ = ['DataflowBenchmark']

class DataflowBenchmark:
    """
    A utility class for benchmarking different dataflow implementations.
    This allows for direct comparison between in-memory and DuckDB-based dataflows.
    """
    
    def __init__(self, benchmark_name: str, output_dir: str = None):
        """
        Initialize a new benchmark.
        
        Args:
            benchmark_name: Name of the benchmark for reporting
            output_dir: Directory to store benchmark results and plots
        """
        self.benchmark_name = benchmark_name
        
        if output_dir is None:
            # Default to a benchmark_results directory in the project root
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'benchmark_results'
            )
        else:
            self.output_dir = output_dir
            
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize result storage
        self.results = []
        self.result_details = {}
        self.process = psutil.Process(os.getpid())
        
    def run_benchmark(self, 
                      name: str, 
                      setup_func: Callable[[], Tuple[Any, Dict[str, Any]]], 
                      execute_func: Callable[[Any, Dict[str, Any]], Any],
                      cleanup_func: Callable[[Any], None] = None,
                      repeat: int = 3) -> Dict[str, Any]:
        """
        Run a benchmark for a specific dataflow implementation.
        
        Args:
            name: Name of the implementation (e.g., "In-Memory", "DuckDB")
            setup_func: Function that sets up the dataflow and returns (dataflow, parameters)
            execute_func: Function that executes the dataflow with (dataflow, parameters)
            cleanup_func: Optional function to clean up after execution
            repeat: Number of times to repeat the benchmark
            
        Returns:
            Dict with benchmark metrics
        """
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
        self.result_details[name] = result_data
        
        print(f"  Average Time: {avg_time:.2f}s, Average Memory: {avg_memory:.2f}MB")
        
        return result_data
    
    def compare_results(self) -> pd.DataFrame:
        """
        Compare benchmark results and return as a DataFrame.
        
        Returns:
            DataFrame with benchmark comparisons
        """
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
    
    def generate_plots(self, save: bool = True) -> Tuple[plt.Figure, plt.Figure]:
        """
        Generate plots comparing the benchmark results.
        
        Args:
            save: Whether to save the plots to files
            
        Returns:
            Two matplotlib figures (time comparison, memory comparison)
        """
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
    
    def save_results(self) -> str:
        """
        Save benchmark results to a CSV file.
        
        Returns:
            Path to the saved CSV file
        """
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
