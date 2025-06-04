from microgrid_demo import MicrogridSetup, run
from src.dataflow.dataflow_factory import DataflowFactory
import time
import sys
import os
import subprocess
# import progress bar
from tqdm import tqdm

def run_single_benchmark(config):
    """Run a single benchmark in a separate process and return the results"""
    # Create Python command to run this benchmark in a fresh interpreter
    cmd = [
        sys.executable, "-c",
        f"""
import sys, os, time
sys.path.insert(0, '{os.getcwd()}')
from microgrid_demo import MicrogridSetup, run
from src.dataflow.dataflow_factory import DataflowFactory

# Configure settings
DataflowFactory.set_execution_mode('{config["mode"]}')
setup = MicrogridSetup()
setup.l = {config["l"]}
setup.T = {config["T"]}
setup.no_homes = {config["no_homes"]}

# Run and get stats
stats = run(setup=setup)
stats["execution_mode"] = '{config["mode"]}'
stats["l"] = {config["l"]}
stats["T"] = {config["T"]}
stats["no_homes"] = {config["no_homes"]}

# Print results as JSON (to capture as output)
import json
print(json.dumps(stats))
        """
    ]
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the JSON output
    try:
        import json
        return json.loads(result.stdout.strip())
    except:
        print(f"Error running benchmark: {result.stderr}")
        print(f"result.stdout: {result.stdout}")
        return None


if __name__ == "__main__":
    start_time = time.time()
    array_l = [1, 4]
    array_T = [24, 48, 96, 168]
    # array_T = [24, 48]
    households = [10, 50, 100, 500]
    # households = [10]
    serial_array = ["serial", "parallel"]
    # serial_array = ["parallel"]
    # Generate all parameter combinations
    benchmark_configs = []
    
    for l in array_l:
        for T in array_T:
            for no_homes in households:
                    for mode in serial_array:
                        benchmark_configs.append({
                            'mode': mode,
                            'l': l,
                            'T': T,
                            'no_homes': no_homes
                        })
    
    # Run benchmarks with a nice progress bar
    stats_array = []
    
    # Create a progress bar with configuration details
    progress_bar = tqdm(
        benchmark_configs,
        desc="Running benchmarks",
        unit="test", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for config in progress_bar:
        # Update progress bar description to show current benchmark
        progress_bar.set_description(
            f"Running: mode={config['mode']}, l={config['l']}, T={config['T']}, homes={config['no_homes']}"
        )
        
        # Run benchmark
        stats = run_single_benchmark(config)
        
        if stats:
            stats_array.append(stats)
            # Also show optimization time in progress bar info
            if "optimizer_time" in stats:
                progress_bar.set_postfix(
                    opt_time=f"{stats['optimizer_time']:.2f}s", 
                    vars=stats.get('variables', 'N/A')
                )
        else:
            progress_bar.set_postfix(status="FAILED")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    
    # Write results to file
    with open("scalability_results.txt", "w") as f:
        for stats in stats_array:
            f.write(str(stats) + "\n")
    
    