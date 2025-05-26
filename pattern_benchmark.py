#!/usr/bin/env python3
"""
Advanced Pattern Benchmark: Tests more complex dataflow patterns to compare
in-memory and DuckDB implementations under different workload scenarios.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager
from src.dataflow.benchmark_patterns.complex_patterns import (
    create_linear_dataflow_pattern,
    create_star_dataflow_pattern,
    create_tree_dataflow_pattern,
    create_cyclic_dataflow_pattern
)
from src.dataflow.benchmark_patterns.pattern_builder import BenchmarkPatternBuilder

class PatternBenchmark:
    """Benchmark class for comparing different dataflow patterns"""
    
    def __init__(self, output_dir='benchmark_results/patterns'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.process = psutil.Process(os.getpid())
        
    def run_benchmark(self, pattern_name, data_size, pattern_func, builder_func):
        """Run benchmark for a specific pattern"""
        print(f"\n=== Running {pattern_name} pattern benchmark with {data_size} rows ===")
        
        # Results storage
        pattern_results = {
            'data_size': data_size,
            'pattern': pattern_name,
            'in_memory': {},
            'duckdb': {},
        }
        
        # Get processing functions
        processing_funcs = pattern_func(data_size)
        
        # Benchmark in-memory
        print("\nTesting In-Memory Implementation:")
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        
        # Setup in-memory dataflow
        builder = BenchmarkPatternBuilder(use_duckdb=False)
        dataflow = builder_func(builder, processing_funcs)
        
        # Execute and time in-memory
        start_time = time.time()
        results_inmem = dataflow.execute({'test': True})
        end_time = time.time()
        
        # Record metrics
        inmem_time = end_time - start_time
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        inmem_memory = final_memory - initial_memory
        
        pattern_results['in_memory'] = {
            'execution_time': inmem_time,
            'memory_usage': inmem_memory,
        }
        
        print(f"Execution time: {inmem_time:.4f}s")
        print(f"Memory usage: {inmem_memory:.2f}MB")
        
        # Cleanup to avoid memory issues
        ParallelExecutionDataFlowManager.getInstance().shutdown()
        ParallelExecutionDataFlowManager._instance = None
        
        # Benchmark DuckDB
        print("\nTesting DuckDB Implementation:")
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        
        # Setup DuckDB dataflow
        builder = BenchmarkPatternBuilder(use_duckdb=True)
        dataflow = builder_func(builder, processing_funcs)
        
        # Execute and time DuckDB
        start_time = time.time()
        results_duckdb = dataflow.execute({'test': True})
        end_time = time.time()
        
        # Record metrics
        duckdb_time = end_time - start_time
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        duckdb_memory = final_memory - initial_memory
        
        pattern_results['duckdb'] = {
            'execution_time': duckdb_time,
            'memory_usage': duckdb_memory
        }
        
        print(f"Execution time: {duckdb_time:.4f}s")
        print(f"Memory usage: {duckdb_memory:.2f}MB")
        
        # Calculate ratios
        time_ratio = duckdb_time / inmem_time if inmem_time > 0 else float('inf')
        memory_ratio = duckdb_memory / inmem_memory if inmem_memory > 0 else float('inf')
        
        pattern_results['ratios'] = {
            'time_ratio': time_ratio,
            'memory_ratio': memory_ratio
        }
        
        print("\n--- Comparison Results ---")
        print(f"Time ratio (DuckDB/In-Memory): {time_ratio:.2f}x")
        print(f"Memory ratio (DuckDB/In-Memory): {memory_ratio:.2f}x")
        
        # Store results
        self.results[f"{pattern_name}_{data_size}"] = pattern_results
        
        # Cleanup
        ParallelExecutionDataFlowManager.getInstance().shutdown()
        ParallelExecutionDataFlowManager._instance = None
        
        return pattern_results
        
    def run_all_patterns(self, data_sizes=None):
        """Run benchmarks for all patterns with different data sizes"""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000]
        
        # Define pattern builders
        patterns = [
            ('Linear', create_linear_dataflow_pattern, 
             lambda builder, funcs: builder.build_linear_pattern(funcs)),
            ('Star', create_star_dataflow_pattern,
             lambda builder, funcs: builder.build_star_pattern(funcs)),
            ('Tree', create_tree_dataflow_pattern,
             lambda builder, funcs: builder.build_tree_pattern(funcs)),
            ('Cyclic', create_cyclic_dataflow_pattern,
             lambda builder, funcs: builder.build_cyclic_pattern(funcs)),
        ]
        
        # Run benchmarks for each pattern and size
        for pattern_name, pattern_func, builder_func in patterns:
            for size in data_sizes:
                self.run_benchmark(pattern_name, size, pattern_func, builder_func)
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report and visualizations"""
        if not self.results:
            print("No results to report")
            return
            
        # Prepare data for plots
        patterns = sorted(set([r.split('_')[0] for r in self.results.keys()]))
        sizes = sorted(set([int(r.split('_')[1]) for r in self.results.keys()]))
        
        # Create figure for time ratios
        plt.figure(figsize=(10, 6))
        
        for pattern in patterns:
            ratios = []
            for size in sizes:
                key = f"{pattern}_{size}"
                if key in self.results:
                    ratios.append(self.results[key]['ratios']['time_ratio'])
                else:
                    ratios.append(float('nan'))
            
            plt.plot(sizes, ratios, 'o-', label=pattern)
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Time Ratio (DuckDB/In-Memory)')
        plt.title('Performance Comparison by Pattern')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pattern_time_ratios.png'))
        
        # Create figure for memory ratios
        plt.figure(figsize=(10, 6))
        
        for pattern in patterns:
            ratios = []
            for size in sizes:
                key = f"{pattern}_{size}"
                if key in self.results:
                    ratios.append(self.results[key]['ratios']['memory_ratio'])
                else:
                    ratios.append(float('nan'))
            
            plt.plot(sizes, ratios, 'o-', label=pattern)
        
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Memory Ratio (DuckDB/In-Memory)')
        plt.title('Memory Usage Comparison by Pattern')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pattern_memory_ratios.png'))
        
        # Save tabular results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.output_dir, f'pattern_results_{timestamp}.csv'), 'w') as f:
            f.write('pattern,data_size,inmem_time,duckdb_time,time_ratio,inmem_memory,duckdb_memory,memory_ratio\n')
            
            for key, result in self.results.items():
                pattern = result['pattern']
                size = result['data_size']
                inmem_time = result['in_memory']['execution_time']
                duckdb_time = result['duckdb']['execution_time']
                time_ratio = result['ratios']['time_ratio']
                inmem_memory = result['in_memory']['memory_usage']
                duckdb_memory = result['duckdb']['memory_usage']
                memory_ratio = result['ratios']['memory_ratio']
                
                f.write(f'{pattern},{size},{inmem_time},{duckdb_time},{time_ratio},{inmem_memory},{duckdb_memory},{memory_ratio}\n')
        
        # Generate summary text
        with open(os.path.join(self.output_dir, f'pattern_summary_{timestamp}.txt'), 'w') as f:
            f.write("=== Dataflow Pattern Benchmark Results ===\n\n")
            
            # Overall averages
            avg_time_ratio = sum([r['ratios']['time_ratio'] for r in self.results.values()]) / len(self.results)
            avg_memory_ratio = sum([r['ratios']['memory_ratio'] for r in self.results.values()]) / len(self.results)
            
            f.write(f"Overall Average Time Ratio (DuckDB/In-Memory): {avg_time_ratio:.2f}x\n")
            f.write(f"Overall Average Memory Ratio (DuckDB/In-Memory): {avg_memory_ratio:.2f}x\n\n")
            
            # By pattern
            f.write("=== Results by Pattern ===\n")
            for pattern in patterns:
                pattern_results = [r for k, r in self.results.items() if k.startswith(f"{pattern}_")]
                if pattern_results:
                    avg_time = sum([r['ratios']['time_ratio'] for r in pattern_results]) / len(pattern_results)
                    avg_mem = sum([r['ratios']['memory_ratio'] for r in pattern_results]) / len(pattern_results)
                    
                    f.write(f"\n{pattern} Pattern:\n")
                    f.write(f"  Average Time Ratio: {avg_time:.2f}x\n")
                    f.write(f"  Average Memory Ratio: {avg_mem:.2f}x\n")
                    f.write("  Detailed Results:\n")
                    
                    for size in sizes:
                        key = f"{pattern}_{size}"
                        if key in self.results:
                            r = self.results[key]
                            f.write(f"    Size {size}: Time={r['ratios']['time_ratio']:.2f}x, Memory={r['ratios']['memory_ratio']:.2f}x\n")
            
            # Analysis
            f.write("\n=== Analysis ===\n")
            
            # Find best/worst patterns for DuckDB
            time_by_pattern = {}
            memory_by_pattern = {}
            
            for pattern in patterns:
                pattern_results = [r for k, r in self.results.items() if k.startswith(f"{pattern}_")]
                if pattern_results:
                    time_by_pattern[pattern] = sum([r['ratios']['time_ratio'] for r in pattern_results]) / len(pattern_results)
                    memory_by_pattern[pattern] = sum([r['ratios']['memory_ratio'] for r in pattern_results]) / len(pattern_results)
            
            if time_by_pattern:
                best_time = min(time_by_pattern.items(), key=lambda x: x[1])
                worst_time = max(time_by_pattern.items(), key=lambda x: x[1])
                
                f.write(f"\nDuckDB performs best on {best_time[0]} pattern for execution time ({best_time[1]:.2f}x)\n")
                f.write(f"DuckDB performs worst on {worst_time[0]} pattern for execution time ({worst_time[1]:.2f}x)\n")
            
            if memory_by_pattern:
                best_memory = min(memory_by_pattern.items(), key=lambda x: x[1])
                worst_memory = max(memory_by_pattern.items(), key=lambda x: x[1])
                
                f.write(f"\nDuckDB performs best on {best_memory[0]} pattern for memory usage ({best_memory[1]:.2f}x)\n")
                f.write(f"DuckDB performs worst on {worst_memory[0]} pattern for memory usage ({worst_memory[1]:.2f}x)\n")
            
            # Recommendations
            f.write("\n=== Recommendations ===\n")
            
            if avg_time_ratio > 1.5:
                f.write("\nDuckDB is significantly slower than in-memory across patterns. ")
                f.write("Consider using in-memory implementation unless persistence is required.\n")
            elif avg_time_ratio < 1.2:
                f.write("\nDuckDB is competitive with in-memory for performance. ")
                f.write("Consider using DuckDB when persistence benefits are valuable.\n")
            else:
                f.write("\nDuckDB shows moderate performance overhead. ")
                f.write("Evaluate based on specific pattern requirements and persistence needs.\n")
                
            # Pattern-specific recommendations
            f.write("\nPattern-Specific Recommendations:\n")
            for pattern, ratio in time_by_pattern.items():
                if ratio < 1.2:
                    f.write(f"- For {pattern} pattern: DuckDB is a good choice (only {ratio:.2f}x slower)\n")
                elif ratio > 2.0:
                    f.write(f"- For {pattern} pattern: Prefer in-memory implementation ({ratio:.2f}x faster)\n")
                else:
                    f.write(f"- For {pattern} pattern: Trade-off between performance and persistence ({ratio:.2f}x slower with DuckDB)\n")

        print(f"\nReport generated in {self.output_dir}")

if __name__ == "__main__":
    # Use smaller sizes for testing
    benchmark = PatternBenchmark()
    benchmark.run_all_patterns(data_sizes=[500, 1000])
    
    print("\n=== Pattern Benchmark Complete ===")
    print(f"Results saved in {benchmark.output_dir}")
