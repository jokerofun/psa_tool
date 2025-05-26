# DuckDB vs. In-Memory Dataflow Benchmark System

This directory contains the benchmark framework results for comparing the performance of DuckDB-based and in-memory implementations of the PSA tool's dataflow system.

## Overview

The benchmark system measures and compares two implementations:

1. **In-Memory Implementation**: Standard implementation where all dataframes are kept in memory.
2. **DuckDB Implementation**: Persistent implementation that stores dataframes in DuckDB tables.

The system measures:
- **Execution Time**: How long each implementation takes to process data
- **Memory Usage**: Peak memory consumption during execution

## Benchmark Structure

The benchmarking framework consists of:

1. **Core Benchmarking Utilities** 
   - `src/dataflow/benchmark_utils.py`: Contains the `DataflowBenchmark` class for running benchmarks and collecting metrics

2. **Benchmark Runner Scripts**
   - `dataflow_benchmark.py`: Basic script for running benchmarks
   - `run_extended_benchmark.py`: Script for running benchmarks with larger datasets
   - `comprehensive_benchmark.py`: Advanced script with detailed analysis and visualization

3. **Result Storage**
   - Results are stored in the `benchmark_results` directory
   - Each benchmark run creates CSV files with raw data and PNG files with visualizations

## Running Benchmarks

To run the benchmarks:

```bash
# Basic benchmark with smaller data sizes
python dataflow_benchmark.py

# Extended benchmark with larger data sizes
python run_extended_benchmark.py 

# Comprehensive benchmark with detailed analysis
python comprehensive_benchmark.py
```

## Sample Pipeline

The benchmark uses a standardized pipeline:
1. **Node A**: Generates a random dataframe of specified size
2. **Node B**: Filters data from Node A 
3. **Node C**: Aggregates data from Node B

This represents a typical ETL workflow that:
- Generates/loads data
- Transforms it through filtering
- Aggregates the results

## Analyzing Results

For each benchmark run, the system generates:

1. **CSV files** with raw benchmark data
2. **PNG visualizations** comparing execution time and memory usage
3. **Text summaries** with analysis and recommendations

Look for patterns in how the relative performance changes as data size increases. Key metrics:
- **Time Ratio**: DuckDB execution time / In-Memory execution time
- **Memory Ratio**: DuckDB memory usage / In-Memory memory usage

## Interpreting Results

When analyzing results, consider:

- **Small Data**: For small datasets, overhead costs may dominate, making differences less meaningful
- **Large Data**: With larger datasets, the true performance characteristics are more clearly visible 
- **Crossover Points**: Look for data sizes where one implementation becomes better than the other
- **Persistence Benefits**: Remember that DuckDB provides persistence benefits not measured in raw performance

## Optimizations

Based on benchmark results, consider:

1. Using the in-memory implementation for smaller datasets and interactive work
2. Using DuckDB for larger datasets that need persistence
3. Hybrid approaches where only certain nodes use DuckDB persistence
