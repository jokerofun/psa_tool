import timeit
from memory_profiler import memory_usage
import statistics

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.helpers.microgrid_setup import MicrogridSetup
import examples.GPOD.gpod_microgrid_solution as gpod_microgrid_demo
import examples.CVXPY.native_cvxpy_microgrid_solution as cvxpy_microgrid_demo
import examples.Pyomo.native_pyomo_microgrid_solution as pyomo_microgrid_demo
import examples.GBOML.gboml_microgrid_solution as gboml_microgrid_demo
from examples.helpers.file_writer import empty, write
from plot_results import plot_comparisons

from pygount import ProjectSummary, SourceAnalysis
from glob import glob

def run_exec_time_experiments(setup:MicrogridSetup, microgrid_file_path="", runs=1):
    empty(setup.output_path)

    demo_exec_time = timeit.timeit(lambda: gpod_microgrid_demo.solve_microgrid_gpod(setup), number=runs)
    cvxpy_exec_time = timeit.timeit(lambda: cvxpy_microgrid_demo.solve_microgrid_cvxpy(setup), number=runs)
    pyomo_exec_time = timeit.timeit(lambda: pyomo_microgrid_demo.solve_microgrid_pyomo(setup), number=runs)
    gboml_exec_time = timeit.timeit(lambda: gboml_microgrid_demo.solve_microgrid_gboml(setup, microgrid_file_path), number=runs)
    
    result = { 
        f"execution_time_T{setup.T}Homes{setup.no_homes}": {
            "GPO-D": round(demo_exec_time / runs, 2), 
            "GBOML": round(gboml_exec_time / runs, 2),
            "CVXPY": round(cvxpy_exec_time / runs, 2),
            "Pyomo": round(pyomo_exec_time / runs, 2)
            }
        }

    write(setup.benchmark_output_path, result)

    return result

def run_memory_usage_experiments(setup: MicrogridSetup, microgrid_file_path="", runs=1):
    # empty(setup.output_path)

    demo_memory_usage = memory_usage((lambda: gpod_microgrid_demo.solve_microgrid_gpod(setup=setup)), max_iterations=runs)
    cvxpy_memory_usage = memory_usage((lambda: cvxpy_microgrid_demo.solve_microgrid_cvxpy(setup)), max_iterations=runs)
    pyomo_memory_usage = memory_usage((lambda: pyomo_microgrid_demo.solve_microgrid_pyomo(setup)), max_iterations=runs)
    gboml_memory_usage = memory_usage((lambda: gboml_microgrid_demo.solve_microgrid_gboml(setup, microgrid_file_path)), max_iterations=runs)
    
    result = { 
        f"memory_usage_T{setup.T}Homes{setup.no_homes}": {
            "GPO-D": round(statistics.mean(demo_memory_usage), 2), 
            "GBOML": round(statistics.mean(gboml_memory_usage), 2),
            "CVXPY": round(statistics.mean(cvxpy_memory_usage), 2),
            "Pyomo": round(statistics.mean(pyomo_memory_usage), 2)
            }
        }
    
    write(setup.benchmark_output_path, result)

    return result

if __name__ == "__main__":
    time_periods = [24*1, 24*2, 24*3, 24*5]
    no_homes = [50, 100, 200, 500, 1000]
    results_exec_time = {}
    results_memory_usage = {}

    for t in time_periods:
        setup = MicrogridSetup()
        setup.T = t
        setup.output_path = f"benchmark/results/output_gboml_T{t}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_T{t}.txt"
        results_exec_time[t] = run_exec_time_experiments(setup=setup, microgrid_file_path=microgrid_file_path)
        results_memory_usage[t] = run_memory_usage_experiments(setup=setup, microgrid_file_path=microgrid_file_path)

    results_exec_time = {}
    results_memory_usage = {}
    for n in no_homes:
        setup = MicrogridSetup()
        setup.no_homes = n
        setup.no_solar_panels = n
        setup.output_path = f"benchmark/results/output_gboml_homes{n}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_homes{n}.txt"
        results_exec_time[n] = run_exec_time_experiments(setup=setup, microgrid_file_path=microgrid_file_path)
        results_memory_usage[n] = run_memory_usage_experiments(setup=setup, microgrid_file_path=microgrid_file_path)
