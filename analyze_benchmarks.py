#!/usr/bin/env python3
"""
Benchmark Visualizer: Tool to analyze and visualize benchmark results from
the DuckDB vs In-Memory dataflow benchmarks.

This script can read multiple benchmark result CSV files and generate
comprehensive visualizations and insights.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime

class BenchmarkVisualizer:
    """Tool to analyze and visualize benchmark results"""
    
    def __init__(self, results_dir='benchmark_results'):
        """Initialize the visualizer with results directory"""
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        
    def load_results(self):
        """Load all benchmark results from CSV files"""
        # Find all results CSV files
        csv_files = glob.glob(os.path.join(self.results_dir, '*_results_*.csv'))
        
        for file_path in csv_files:
            try:
                # Extract the data size from the filename
                match = re.search(r'Size_(\d+)_', file_path)
                if match:
                    data_size = int(match.group(1))
                    
                    # Load the data
                    df = pd.read_csv(file_path)
                    
                    # Add data size as a column if not already present
                    if 'data_size' not in df.columns:
                        df['data_size'] = data_size
                    
                    self.results.append(df)
                    print(f"Loaded benchmark data from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Combine all results
        if self.results:
            self.combined_results = pd.concat(self.results, ignore_index=True)
            print(f"Combined {len(self.results)} benchmark results")
            return self.combined_results
        else:
            print("No benchmark results were found")
            return None

    def generate_visualizations(self):
        """Generate comprehensive visualizations from results"""
        if not hasattr(self, 'combined_results') or self.combined_results is None:
            print("No results to visualize. Run load_results() first.")
            return
            
        # Filter to just in-memory and DuckDB implementations
        df = self.combined_results
        
        # Extract data sizes and sort them
        in_memory_mask = df['name'].str.contains('In-Memory')
        duckdb_mask = df['name'].str.contains('DuckDB')
        
        if not (in_memory_mask.any() and duckdb_mask.any()):
            print("Cannot find both In-Memory and DuckDB results for comparison")
            return
            
        in_memory_results = df[in_memory_mask].copy()
        duckdb_results = df[duckdb_mask].copy()
        
        # Extract data sizes
        in_memory_results['data_size'] = in_memory_results['name'].str.extract(r'In-Memory_(\d+)').astype(int)
        duckdb_results['data_size'] = duckdb_results['name'].str.extract(r'DuckDB_(\d+)').astype(int)
        
        # Create merged dataframe with comparison metrics
        merged = pd.DataFrame()
        merged['data_size'] = sorted(in_memory_results['data_size'].unique())
        
        # Add execution times
        for size in merged['data_size']:
            inmem_time = in_memory_results[in_memory_results['data_size'] == size]['avg_execution_time'].values
            duckdb_time = duckdb_results[duckdb_results['data_size'] == size]['avg_execution_time'].values
            
            if len(inmem_time) > 0 and len(duckdb_time) > 0:
                merged.loc[merged['data_size'] == size, 'in_memory_time'] = inmem_time[0]
                merged.loc[merged['data_size'] == size, 'duckdb_time'] = duckdb_time[0]
                merged.loc[merged['data_size'] == size, 'time_ratio'] = duckdb_time[0] / inmem_time[0]
        
        # Add memory usage
        for size in merged['data_size']:
            inmem_mem = in_memory_results[in_memory_results['data_size'] == size]['avg_memory_usage'].values
            duckdb_mem = duckdb_results[duckdb_results['data_size'] == size]['avg_memory_usage'].values
            
            if len(inmem_mem) > 0 and len(duckdb_mem) > 0:
                merged.loc[merged['data_size'] == size, 'in_memory_memory'] = inmem_mem[0]
                merged.loc[merged['data_size'] == size, 'duckdb_memory'] = duckdb_mem[0]
                merged.loc[merged['data_size'] == size, 'memory_ratio'] = duckdb_mem[0] / inmem_mem[0]
        
        # Save the merged results
        merged.to_csv(os.path.join(self.output_dir, 'merged_comparison.csv'), index=False)
        
        # Generate visualizations
        
        # 1. Execution Time Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(merged['data_size'], merged['in_memory_time'], 'o-', label='In-Memory')
        plt.plot(merged['data_size'], merged['duckdb_time'], 's-', label='DuckDB')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'execution_time_comparison.png'))
        
        # 2. Memory Usage Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(merged['data_size'], merged['in_memory_memory'], 'o-', label='In-Memory')
        plt.plot(merged['data_size'], merged['duckdb_memory'], 's-', label='DuckDB')
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'memory_usage_comparison.png'))
        
        # 3. Performance Ratio (DuckDB/In-Memory)
        plt.figure(figsize=(10, 6))
        plt.plot(merged['data_size'], merged['time_ratio'], 'o-', color='red')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Time Ratio (DuckDB / In-Memory)')
        plt.title('Performance Ratio (DuckDB vs In-Memory)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'performance_ratio.png'))
        
        # 4. Memory Ratio (DuckDB/In-Memory)
        plt.figure(figsize=(10, 6))
        plt.plot(merged['data_size'], merged['memory_ratio'], 'o-', color='green')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Memory Ratio (DuckDB / In-Memory)')
        plt.title('Memory Usage Ratio (DuckDB vs In-Memory)')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'memory_ratio.png'))
        
        # 5. Combined Log Scale Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.plot(merged['data_size'], merged['in_memory_time'], 'o-', label='In-Memory')
        ax1.plot(merged['data_size'], merged['duckdb_time'], 's-', label='DuckDB')
        ax1.set_xlabel('Data Size (rows)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison (Log Scale)')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(merged['data_size'], merged['in_memory_memory'], 'o-', label='In-Memory')
        ax2.plot(merged['data_size'], merged['duckdb_memory'], 's-', label='DuckDB')
        ax2.set_xlabel('Data Size (rows)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison_log_scale.png'))
        
        # Generate a text summary
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write("=== Benchmark Analysis Summary ===\n\n")
            
            f.write("--- Performance by Data Size ---\n")
            for _, row in merged.iterrows():
                f.write(f"\nData Size: {row['data_size']} rows\n")
                f.write(f"  In-Memory Execution Time: {row.get('in_memory_time', 'N/A'):.4f}s\n")
                f.write(f"  DuckDB Execution Time: {row.get('duckdb_time', 'N/A'):.4f}s\n")
                f.write(f"  Time Ratio (DuckDB/In-Memory): {row.get('time_ratio', 'N/A'):.2f}x\n")
                f.write(f"  In-Memory Memory Usage: {row.get('in_memory_memory', 'N/A'):.2f}MB\n")
                f.write(f"  DuckDB Memory Usage: {row.get('duckdb_memory', 'N/A'):.2f}MB\n")
                f.write(f"  Memory Ratio (DuckDB/In-Memory): {row.get('memory_ratio', 'N/A'):.2f}x\n")
            
            # General conclusions
            f.write("\n--- Overall Findings ---\n")
            
            # Time trends
            avg_time_ratio = merged['time_ratio'].mean()
            f.write(f"On average, DuckDB is {avg_time_ratio:.2f}x ")
            f.write("slower than" if avg_time_ratio > 1 else "faster than")
            f.write(" In-Memory for execution time.\n")
            
            # Check if the ratio changes with data size
            if len(merged) > 1:
                time_ratio_trend = merged['time_ratio'].iloc[-1] - merged['time_ratio'].iloc[0]
                if abs(time_ratio_trend) > 0.1:  # Significant trend
                    f.write(f"The performance gap {'increases' if time_ratio_trend > 0 else 'decreases'} ")
                    f.write(f"with larger data sizes (change of {time_ratio_trend:.2f}x).\n")
            
            # Memory trends
            avg_memory_ratio = merged['memory_ratio'].mean()
            f.write(f"On average, DuckDB uses {avg_memory_ratio:.2f}x ")
            f.write("more memory than" if avg_memory_ratio > 1 else "less memory than")
            f.write(" In-Memory.\n")
            
            # Check if the ratio changes with data size
            if len(merged) > 1:
                memory_ratio_trend = merged['memory_ratio'].iloc[-1] - merged['memory_ratio'].iloc[0]
                if abs(memory_ratio_trend) > 0.1:  # Significant trend
                    f.write(f"The memory usage difference {'increases' if memory_ratio_trend > 0 else 'decreases'} ")
                    f.write(f"with larger data sizes (change of {memory_ratio_trend:.2f}x).\n")
            
            # Recommendations
            f.write("\n--- Recommendations ---\n")
            if avg_time_ratio <= 1.2 and avg_memory_ratio <= 1.2:
                f.write("Both implementations perform similarly. Choose based on other factors like persistence needs.\n")
            elif avg_time_ratio <= 1.2 and avg_memory_ratio > 1.2:
                f.write("In-Memory has memory advantages with similar execution time. Prefer In-Memory unless persistence is needed.\n")
            elif avg_time_ratio > 1.2 and avg_memory_ratio <= 1.2:
                f.write("In-Memory has speed advantages with similar memory usage. Prefer In-Memory unless persistence is needed.\n")
            else:
                f.write("In-Memory performs better in both time and memory. Only use DuckDB when persistence is required.\n")
                
        print(f"Visualizations and analysis saved to: {self.output_dir}")
        return merged

if __name__ == '__main__':
    print("Benchmark Visualizer - Analyzing DuckDB vs In-Memory Performance")
    
    # Create visualizer and process results
    visualizer = BenchmarkVisualizer()
    visualizer.load_results()
    visualizer.generate_visualizations()
    
    print("Analysis complete!")
