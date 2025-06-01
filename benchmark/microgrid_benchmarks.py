
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

def plot_execution_time(results={}):
    import matplotlib.pyplot as plt

    plt.bar(results.keys(), results.values(), color='skyblue')

    plt.xlabel("Implementations using")
    plt.ylabel("Average execution time in seconds")
    plt.title("Average Execution Time for Different Microgrid Implementations")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig("figures/execution_time_comparison.png")
    plt.show()


def measure_execution_time():
    setup = MicrogridSetup()
    gboml_setup = MicrogridSetup()
    gboml_setup.T = 25

    benchmark = Benchmark()
    empty(setup.output_path)
    gboml_result = benchmark.run(gboml_microgrid_demo.run, gboml_setup, runs=1)
    demo_result = benchmark.run(microgrid_demo.run, setup, runs=1)
    cvxpy_result = benchmark.run(cvxpy_microgrid_demo.solve_microgrid, setup, runs=1)
    pyomo_result = benchmark.run(pyomo_microgrid_demo.solve_microgrid_pyomo, setup, runs=1)

    plot_execution_time({"demo": demo_result["average_time"], 
                         "GBOML": gboml_result["average_time"],
                         "CVXPY": cvxpy_result["average_time"],
                         "Pyomo": pyomo_result["average_time"]
                         })

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
    # measure_execution_time()
    measure_lines_of_code(folders=["GBOML", "Pyomo", "CVXPY"])
