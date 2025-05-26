#!/usr/bin/env python3
"""
Extended Dataflow Benchmark Script: Compares in-memory vs. DuckDB dataflow implementations
with larger datasets to better demonstrate performance differences.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the benchmark functions
from dataflow_benchmark import run_benchmark

if __name__ == "__main__":
    # Use larger data sizes for more comprehensive benchmarks
    data_sizes = [10000, 50000, 100000, 250000]
    run_benchmark(data_sizes=data_sizes, repeat=3)
    
    print("\n=== Extended Benchmark Complete ===")
    print(f"Results saved in benchmark_results directory")
