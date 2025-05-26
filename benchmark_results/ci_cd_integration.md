# Integrating Dataflow Benchmarks into CI/CD Pipeline

This document outlines how to integrate the dataflow benchmarking system into a CI/CD pipeline to regularly monitor performance and detect regressions.

## Overview

By integrating benchmark tests into a CI/CD pipeline, we can:
1. Continuously monitor performance metrics
2. Detect regressions early in the development process
3. Create historical performance data for trend analysis
4. Automate performance reporting

## Implementation Steps

### 1. Set Up Benchmark Job

Configure a CI job (e.g., in GitHub Actions, GitLab CI, or Jenkins) to run the benchmarks:

```yaml
# Example GitHub Actions workflow
name: Dataflow Performance Benchmark

on:
  # Run on specific branches, e.g. when merging to main
  push:
    branches: [main, develop]
  
  # Run on PRs to main or develop
  pull_request:
    branches: [main, develop]
    
  # Schedule regular runs (e.g., daily)
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight UTC

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
      - name: Run benchmarks
        run: python dataflow_benchmark.py --ci-mode
        
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

### 2. Create CI-Specific Benchmark Mode

Add a special mode for CI benchmarks that:
- Runs with smaller data sizes to complete quickly (default)
- Can optionally run full benchmarks on scheduled runs
- Outputs results in a CI-friendly format

Example modifications to the benchmark script:

```python
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run dataflow benchmarks')
    parser.add_argument('--ci-mode', action='store_true', 
                        help='Run in CI mode with reduced data sizes')
    parser.add_argument('--full', action='store_true',
                        help='In CI mode, run full benchmarks instead of reduced')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.ci_mode:
        if args.full:
            # Full benchmark for scheduled runs
            data_sizes = [1000, 10000, 50000, 100000]
            repeat = 3
        else:
            # Quick benchmark for PR checks
            data_sizes = [1000, 5000]
            repeat = 2
            
        # Use JUnit format for CI integration
        output_format = "junit"
    else:
        # Default for local runs
        data_sizes = [1000, 5000, 10000, 25000, 50000]
        repeat = 3
        output_format = "standard"
        
    run_benchmark(data_sizes=data_sizes, repeat=repeat, output_format=output_format)
```

### 3. Add Performance Regression Detection

Implement regression detection to alert when performance degrades:

```python
def check_for_regressions(new_results, baseline_file="benchmark_baseline.json"):
    """Check if current results show regression compared to baseline."""
    # Define threshold for flagging regression (e.g., 10%)
    REGRESSION_THRESHOLD = 1.1
    
    # Load baseline results
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    else:
        print("No baseline found, skipping regression check")
        return False
    
    # Compare with current results
    has_regression = False
    for size in new_results:
        if str(size) in baseline:
            # Check execution time regression
            if (new_results[size]["in_memory_time"] > 
                    baseline[str(size)]["in_memory_time"] * REGRESSION_THRESHOLD):
                print(f"REGRESSION: In-memory execution time increased by "
                      f"{new_results[size]['in_memory_time']/baseline[str(size)]['in_memory_time']:.2f}x "
                      f"for data size {size}")
                has_regression = True
                
            # Check DuckDB time regression
            if (new_results[size]["duckdb_time"] > 
                    baseline[str(size)]["duckdb_time"] * REGRESSION_THRESHOLD):
                print(f"REGRESSION: DuckDB execution time increased by "
                      f"{new_results[size]['duckdb_time']/baseline[str(size)]['duckdb_time']:.2f}x "
                      f"for data size {size}")
                has_regression = True
    
    return has_regression
```

### 4. Implement Historical Trend Tracking

Store historical benchmark results to track performance over time:

```python
def store_historical_data(results):
    """Store benchmark results in historical database."""
    history_dir = "benchmark_history"
    os.makedirs(history_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(history_dir, f"benchmark_{timestamp}.json")
    
    # Store results
    with open(history_file, 'w') as f:
        json.dump(results, f)
    
    # Update latest results
    with open(os.path.join(history_dir, "latest.json"), 'w') as f:
        json.dump(results, f)
```

### 5. Generate Performance Reports

Create a report generation script that can be run after benchmarks:

```python
def generate_performance_report():
    """Generate a performance report based on historical data."""
    # Load historical data
    history_files = sorted(glob.glob("benchmark_history/benchmark_*.json"))
    if not history_files:
        print("No historical data found")
        return
    
    # Process the data
    history = []
    for file in history_files[-30:]:  # Last 30 results
        with open(file, 'r') as f:
            data = json.load(f)
            timestamp = os.path.basename(file).split("_")[1]
            history.append({"timestamp": timestamp, "data": data})
    
    # Generate report
    with open("performance_report.md", "w") as f:
        f.write("# Dataflow Performance Trend Report\n\n")
        f.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add relevant sections with plots and tables
        # ...
```

### 6. Publish Results as GitHub Pages

Set up automatic publishing of benchmark results:

```yaml
# In CI workflow
      - name: Generate performance report
        run: python generate_performance_report.py
        
      - name: Publish performance report
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./performance_reports
          destination_dir: performance
```

## Additional Considerations

1. **Stable Testing Environment:** Ensure CI runners have consistent resources to avoid benchmark variability.

2. **Separate Workflow:** Consider running benchmarks in a separate workflow to avoid delaying the main CI pipeline.

3. **Scheduled Full Benchmarks:** Run quick benchmarks on PRs for immediate feedback, but schedule full benchmarks daily or weekly.

4. **Alert Mechanisms:** Set up notifications when significant performance regressions are detected.

5. **Baseline Updates:** Periodically update baseline results when legitimate performance changes are expected.

## Examples 

### Example: PR Comment with Performance Comparison

You can set up the CI to comment on PRs with benchmark results:

```python
def generate_pr_comment(results, baseline):
    """Generate a comment for PR with performance comparison."""
    comment = "## Performance Benchmark Results\n\n"
    comment += "| Data Size | In-Memory Δ | DuckDB Δ |\n"
    comment += "| --- | --- | --- |\n"
    
    for size in results:
        if str(size) in baseline:
            in_mem_change = results[size]["in_memory_time"] / baseline[str(size)]["in_memory_time"]
            duckdb_change = results[size]["duckdb_time"] / baseline[str(size)]["duckdb_time"]
            
            in_mem_emoji = "🟢" if in_mem_change < 0.95 else "🟡" if in_mem_change < 1.05 else "🔴"
            duckdb_emoji = "🟢" if duckdb_change < 0.95 else "🟡" if duckdb_change < 1.05 else "🔴"
            
            comment += f"| {size} | {in_mem_emoji} {in_mem_change:.2f}x | {duckdb_emoji} {duckdb_change:.2f}x |\n"
    
    return comment
```

### Example: Performance Dashboard

Create a dashboard that shows performance trends over time:

```python
def generate_dashboard():
    """Generate an HTML performance dashboard."""
    # Load historical data
    # Generate plots with matplotlib or plotly
    # Create an HTML dashboard
    # ...
```

By integrating these benchmarks into the CI/CD pipeline, you can ensure that performance regressions are caught early and that the team has visibility into the performance characteristics of both implementations.
