
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import microgrid_demo
from examples.helpers.microgrid_setup import MicrogridSetup
import examples.CVXPY.native_cvxpy_microgrid_solution as cvxpy_microgrid_demo
import examples.Pyomo.native_pyomo_microgrid_solution as pyomo_microgrid_demo
import examples.GBOML.gboml_microgrid_solution as gboml_microgrid_demo
from examples.helpers.file_writer import empty
from benchmark import Benchmark

from pygount import ProjectSummary, SourceAnalysis
from glob import glob

from plot_results import plot_comparisons

def measure_execution_time(setup:MicrogridSetup, microgrid_file_path):
    benchmark = Benchmark()
    empty(setup.output_path)
    demo_result = benchmark.run(microgrid_demo.run, setup, runs=1)
    cvxpy_result = benchmark.run(cvxpy_microgrid_demo.solve_microgrid_cvxpy, setup, runs=1)
    pyomo_result = benchmark.run(pyomo_microgrid_demo.solve_microgrid_pyomo, setup, runs=1)
    gboml_result = benchmark.run(gboml_microgrid_demo.run, setup, microgrid_file_path, runs=1)

    return {
            "demo": demo_result["average_time"], 
            "GBOML": gboml_result["average_time"],
            "CVXPY": cvxpy_result["average_time"],
            "Pyomo": pyomo_result["average_time"]
        }

def measure_lines_of_code(folders):
    for folder in folders:
        project_summary = ProjectSummary()
        source_paths = glob(f"examples/{folder}/*.py") + glob(f"examples/{folder}/*.txt")
        for source_path in source_paths:
            source_analysis = SourceAnalysis.from_file(source_path, "gboml_test")
            project_summary.add(source_analysis)

        print(folder)
        print("-"*100)
        print(f"Code Count: {project_summary.total_code_count}")
        print(f"Documentation Count: {project_summary.total_documentation_count}")
        print(f"Empty Lines Count: {project_summary.total_empty_count}")
        print(f"Total: {project_summary._total_line_count}")
        print("-"*100)
        for language_summary in project_summary.language_to_language_summary_map.values():
            print(language_summary)
        print("-"*100)

if __name__ == "__main__":
    time_periods = [24, 48, 72]
    no_homes = [50, 100, 200]
    results = {}

    for t in time_periods:
        setup = MicrogridSetup()
        setup.T = t
        setup.output_path = f"figures/output_T{t}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_T{t}.txt"
        results[t] = measure_execution_time(setup=setup, microgrid_file_path=microgrid_file_path)

    plot_comparisons(results=results, x_label="Time length in hours", output_path="figures/exec_time_comparison_scaleT.png")

    results = {}
    for n in no_homes:
        setup = MicrogridSetup()
        setup.no_homes = n
        setup.no_solar_panels = n
        setup.output_path = f"figures/output_homes{n}.txt"
        microgrid_file_path = f"examples/GBOML/microgrid_homes{n}.txt"
        results[n] = measure_execution_time(setup=setup, microgrid_file_path=microgrid_file_path)

    plot_comparisons(results=results, x_label="Number of homes and solar panels", output_path="figures/exec_time_comparison_scaleHomes.png")
    # measure_lines_of_code(folders=["GBOML", "Pyomo", "CVXPY"])
