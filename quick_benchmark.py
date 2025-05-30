#!/usr/bin/env python3
"""
Quick Benchmark: Performs smaller but more varied data size benchmarks for faster testing

This script runs benchmarks with more intermediate data sizes to better show trends
and crossover points between in-memory and DuckDB implementations.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataflow_benchmark import run_benchmark

if __name__ == "__main__":
    # Use a wider range of data sizes with smaller step sizes
    data_sizes = [500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]
    
    # Run with less repetition for faster results
    run_benchmark(data_sizes=data_sizes, repeat=2)
    
    print("\n=== Quick Benchmark Complete ===")
    print(f"Results saved in benchmark_results directory")
