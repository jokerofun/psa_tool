import timeit
from memory_profiler import memory_usage
import statistics

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.helpers.microgrid_setup import MicrogridSetup
import examples.our_tool.microgrid_demo as microgrid_demo
import examples.CVXPY.native_cvxpy_microgrid_solution as cvxpy_microgrid_demo
import examples.Pyomo.native_pyomo_microgrid_solution as pyomo_microgrid_demo
import examples.GBOML.gboml_microgrid_solution as gboml_microgrid_demo
from examples.helpers.file_writer import empty, write
from benchmark import Benchmark
from plot_results import plot_comparisons

from pygount import ProjectSummary, SourceAnalysis
from glob import glob

def run_exec_time_experiments(setup:MicrogridSetup, microgrid_file_path="", runs=1):
    empty(setup.output_path)

    demo_exec_time = timeit.timeit(lambda: microgrid_demo.solve_microgrid(setup), number=runs)
    cvxpy_exec_time = timeit.timeit(lambda: pyomo_microgrid_demo.solve_microgrid_pyomo(setup), number=runs)
    pyomo_exec_time = timeit.timeit(lambda: pyomo_microgrid_demo.solve_microgrid_pyomo(setup), number=runs)
    gboml_exec_time = timeit.timeit(lambda: gboml_microgrid_demo.run(setup, microgrid_file_path), number=runs)
    
    result = { 
        "execution time": {
            "demo": round(demo_exec_time / runs, 2), 
            "GBOML": round(gboml_exec_time / runs, 2),
            "CVXPY": round(cvxpy_exec_time / runs, 2),
            "Pyomo": round(pyomo_exec_time / runs, 2)
            }
        }

    write(setup.benchmark_output_path, result)

    return result

def run_memory_usage_experiments(setup: MicrogridSetup, microgrid_file_path="", runs=1):
    empty(setup.output_path)

    demo_memory_usage = memory_usage((lambda: microgrid_demo.run(setup=setup)), max_iterations=runs)
    cvxpy_memory_usage = memory_usage((lambda: cvxpy_microgrid_demo.solve_microgrid_cvxpy(setup)), max_iterations=runs)
    pyomo_memory_usage = memory_usage((lambda: pyomo_microgrid_demo.solve_microgrid_pyomo(setup)), max_iterations=runs)
    gboml_memory_usage = memory_usage((lambda: gboml_microgrid_demo.run(setup, microgrid_file_path)), max_iterations=runs)
    
    result = { 
        "memory usage": {
            "demo": round(statistics.mean(demo_memory_usage), 2), 
            "GBOML": round(statistics.mean(gboml_memory_usage), 2),
            "CVXPY": round(statistics.mean(cvxpy_memory_usage), 2),
            "Pyomo": round(statistics.mean(pyomo_memory_usage), 2)
            }
        }
    
    write(setup.benchmark_output_path, result)

    return result

if __name__ == "__main__":
    time_periods = [24, 48, 72]
    no_homes = [50, 100, 200]
    results_exec_time = {}
    results_memory_usage = {}

    for t in time_periods:
        setup = MicrogridSetup()
        setup.T = t
        setup.output_path = f"figures/output_T{t}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_T{t}.txt"
        results_exec_time[t] = run_exec_time_experiments(setup=setup, microgrid_file_path=microgrid_file_path)
        results_memory_usage[t] = run_memory_usage_experiments(setup=setup, microgrid_file_path=microgrid_file_path)

    plot_comparisons(results=results_exec_time,
                     title="Average execution time for different microgrid implementations", 
                     x_label="Time length for microgrid problem in hours", 
                     y_label="Average execution time in seconds", 
                     output_path="figures/exec_time_comparison_scaleT.png")
    plot_comparisons(results=results_memory_usage, 
                     title="Average memory usage for different microgrid implementations",
                     x_label="Time length for microgrid problem in hours",
                     y_label="Average memory usage in MiB",
                     output_path="figures/memory_usage_comparison_scaleT.png")

    results_exec_time = {}
    results_memory_usage = {}
    for n in no_homes:
        setup = MicrogridSetup()
        setup.no_homes = n
        setup.no_solar_panels = n
        setup.output_path = f"figures/output_homes{n}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_homes{n}.txt"
        results_exec_time[n] = run_exec_time_experiments(setup=setup, microgrid_file_path=microgrid_file_path)
        results_memory_usage[n] = run_memory_usage_experiments(setup=setup, microgrid_file_path=microgrid_file_path)


    plot_comparisons(results=results_exec_time,
                     title="Average execution time for different microgrid implementations", 
                     x_label="Number of homes and solar panels for microgrid problem", 
                     y_label="Average execution time in seconds", 
                     output_path="figures/exec_time_comparison_scaleHomes.png")
    plot_comparisons(results=results_memory_usage, 
                     title="Average memory usage for different microgrid implementations",
                     x_label="Number of homes and solar panels for microgrid problem",
                     y_label="Average memory usage in MiB",
                     output_path="figures/memory_usage_comparison_scaleHomes.png")
    # measure_lines_of_code(folders=["GBOML", "Pyomo", "CVXPY"])
