#!/usr/bin/env python3
"""
Memory Analysis: Analysis of memory usage patterns in DuckDB and in-memory dataflow implementations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
import numpy as np

def analyze_memory_patterns(csv_files=None, output_dir='benchmark_results/memory_analysis'):
    """
    Analyze memory usage patterns from benchmark results.
    
    Args:
        csv_files: List of CSV files with benchmark results
        output_dir: Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If no files provided, look for them in the benchmark_results directory
    if csv_files is None:
        benchmark_dir = 'benchmark_results'
        csv_files = []
        for root, _, files in os.walk(benchmark_dir):
            for file in files:
                if file.endswith('.csv') and 'results' in file and not file.startswith('merged'):
                    csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("No benchmark result files found.")
        return
    
    # Load and combine data from all CSV files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add source filename
            df['source'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid data found in CSV files.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if this is old or new format data
    if 'name' in combined_df.columns:
        # Old format - extract implementation and size from name column
        combined_df[['implementation', 'size']] = combined_df['name'].str.split('_', expand=True)
        combined_df['data_size'] = combined_df['size'].astype(int)
    
    # Save combined data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_df.to_csv(os.path.join(output_dir, f'combined_results_{timestamp}.csv'), index=False)
    
    # Group by data size
    grouped = combined_df.groupby('data_size')
    
    # Create data for explanation plots
    sizes = []
    inmem_mem = []
    duckdb_mem = []
    ratios = []
    
    for size, group in grouped:
        # For each size, get in-memory and DuckDB measurements
        inmem_group = group[group['implementation'] == 'In-Memory'] if 'implementation' in group else group
        duckdb_group = group[group['implementation'] == 'DuckDB'] if 'implementation' in group else group
        
        # Get average memory usage
        if 'avg_memory_usage' in group:
            avg_inmem = inmem_group['avg_memory_usage'].mean() if not inmem_group.empty else 0
            avg_duckdb = duckdb_group['avg_memory_usage'].mean() if not duckdb_group.empty else 0
        else:
            avg_inmem = inmem_group['inmem_memory'].mean() if 'inmem_memory' in inmem_group else 0
            avg_duckdb = duckdb_group['duckdb_memory'].mean() if 'duckdb_memory' in duckdb_group else 0
        
        # Calculate ratio
        ratio = avg_duckdb / avg_inmem if avg_inmem > 0 else 0
        
        sizes.append(size)
        inmem_mem.append(avg_inmem)
        duckdb_mem.append(avg_duckdb)
        ratios.append(ratio)
    
    # Sort by size
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    inmem_mem = [inmem_mem[i] for i in sorted_indices]
    duckdb_mem = [duckdb_mem[i] for i in sorted_indices]
    ratios = [ratios[i] for i in sorted_indices]
    
    # Create memory usage plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, inmem_mem, 'o-', label='In-Memory')
    plt.plot(sizes, duckdb_mem, 's-', label='DuckDB')
    plt.xlabel('Data Size (rows)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'memory_usage_{timestamp}.png'))
    
    # Create ratio plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, ratios, 'o-', color='green')
    plt.axhline(y=1.0, linestyle='--', color='gray')
    plt.xlabel('Data Size (rows)')
    plt.ylabel('Memory Ratio (DuckDB/In-Memory)')
    plt.title('Memory Usage Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'memory_ratio_{timestamp}.png'))
    
    # Create log scale plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(sizes, inmem_mem, 'o-', label='In-Memory')
    plt.semilogy(sizes, duckdb_mem, 's-', label='DuckDB')
    plt.xlabel('Data Size (rows)')
    plt.ylabel('Memory Usage (MB) - Log Scale')
    plt.title('Memory Usage Comparison (Log Scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'memory_log_scale_{timestamp}.png'))
    
    # Generate explanation text
    with open(os.path.join(output_dir, f'memory_analysis_{timestamp}.txt'), 'w') as f:
        f.write("=== Memory Usage Pattern Analysis ===\n\n")
        
        # General observations
        f.write("General Observations:\n")
        f.write(f"- Data sizes analyzed: {', '.join(map(str, sizes))}\n")
        
        # DuckDB memory pattern
        duckdb_trend = "increases" if duckdb_mem[-1] > duckdb_mem[0] else "decreases"
        f.write(f"\nDuckDB Memory Pattern:\n")
        f.write(f"- DuckDB memory usage {duckdb_trend} with larger data sizes\n")
        f.write(f"- Memory range: {min(duckdb_mem):.2f}MB to {max(duckdb_mem):.2f}MB\n")
        
        # In-Memory pattern
        inmem_trend = "increases" if inmem_mem[-1] > inmem_mem[0] else "decreases"
        f.write(f"\nIn-Memory Pattern:\n")
        f.write(f"- In-memory implementation memory usage {inmem_trend} with larger data sizes\n")
        f.write(f"- Memory range: {min(inmem_mem):.2f}MB to {max(inmem_mem):.2f}MB\n")
        
        # Ratio analysis
        f.write(f"\nMemory Ratio Analysis:\n")
        if all(r > 1 for r in ratios):
            f.write("- DuckDB consistently uses more memory than in-memory implementation\n")
        elif all(r < 1 for r in ratios):
            f.write("- DuckDB consistently uses less memory than in-memory implementation\n")
        else:
            crossover_points = [i for i in range(1, len(ratios)) if (ratios[i-1] < 1 and ratios[i] > 1) or (ratios[i-1] > 1 and ratios[i] < 1)]
            if crossover_points:
                crossover_sizes = [sizes[i] for i in crossover_points]
                f.write(f"- Crossover points detected at data sizes: {', '.join(map(str, crossover_sizes))}\n")
                f.write("- DuckDB's memory efficiency relative to in-memory implementation changes based on data size\n")
        
        # Memory efficiency pattern
        ratio_trend = "increases" if ratios[-1] > ratios[0] else "decreases"
        f.write(f"- DuckDB/In-Memory memory ratio {ratio_trend} with larger data sizes\n")
        f.write(f"- Ratio range: {min(ratios):.2f}x to {max(ratios):.2f}x\n")
        
        # Potential explanation
        f.write("\nPotential Explanation for Memory Usage Patterns:\n")
        
        if any(r < 1 for r in ratios):
            f.write("1. DuckDB shows lower memory usage for some data sizes because:\n")
            f.write("   - It may use more efficient data structures for larger datasets\n")
            f.write("   - It may implement memory paging/spilling to disk for larger data\n")
            f.write("   - It has specialized storage formats optimized for columnar data\n")
        
        if any(r > 1 for r in ratios):
            f.write("2. DuckDB shows higher memory usage for some data sizes because:\n")
            f.write("   - It maintains additional metadata for query optimization\n")
            f.write("   - It pre-allocates memory buffers that may be underutilized for small datasets\n")
            f.write("   - The overhead of DuckDB's persistence layer adds memory requirements\n")
        
        # Inconsistent measurements
        if max(ratios) / min(ratios) > 3:
            f.write("\n3. The large variation in memory ratios suggests:\n")
            f.write("   - Memory measurement methods may be inconsistent or affected by garbage collection timing\n")
            f.write("   - Some measurements may include temporary allocations not present in others\n")
            f.write("   - The benchmark environment may have background processes affecting measurements\n")
        
        # Recommendations
        f.write("\nRecommendations:\n")
        f.write("1. Use process isolation and consistent measurement techniques for more reliable results\n")
        f.write("2. Consider peak memory usage rather than average for critical applications\n")
        f.write("3. Choose implementation based on your typical data size range\n")
        
    print(f"\nAnalysis results saved to {output_dir}")
    return combined_df

if __name__ == "__main__":
    # If files are provided as command line arguments, use them
    files = sys.argv[1:] if len(sys.argv) > 1 else None
    analyze_memory_patterns(files)
