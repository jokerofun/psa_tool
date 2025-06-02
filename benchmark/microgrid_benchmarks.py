
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

# def plot_execution_time(results={}):
#     import matplotlib.pyplot as plt

#     plt.bar(results.keys(), results.values(), color='skyblue')

#     plt.xlabel("Implementations using")
#     plt.ylabel("Average execution time in seconds")
#     plt.title("Average Execution Time for Different Microgrid Implementations")
#     plt.grid(axis="y", linestyle="--", alpha=0.7)

#     for i, v in enumerate(results.values()):
#         plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

#     plt.tight_layout()
#     plt.savefig("figures/execution_time_comparison.png")
#     plt.show()

def plot_execution_time(results={}):
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten results: keys are (scenario, implementation), values are execution times
    flat_results = {}
    for scenario, impl_times in results.items():
        for impl, time in impl_times.items():
            flat_results[(str(scenario), impl)] = time

    scenarios = sorted(set(k[0] for k in flat_results.keys()))
    implementations = sorted(set(k[1] for k in flat_results.keys()))

    # Prepare data for grouped bar plot
    bar_width = 0.2
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, impl in enumerate(implementations):
        times = [flat_results.get((str(scenario), impl), 0) for scenario in scenarios]
        ax.bar(x + idx * bar_width, times, width=bar_width, label=impl)

    ax.set_xlabel("Scenario (T or no_homes)")
    ax.set_ylabel("Average execution time in seconds")
    ax.set_title("Average Execution Time for Different Microgrid Implementations")
    ax.set_xticks(x + bar_width * (len(implementations)-1)/2)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/execution_time_comparison.png")
    plt.show()

def measure_execution_time(setup:MicrogridSetup):
    benchmark = Benchmark()
    empty(setup.output_path)
    demo_result = benchmark.run(microgrid_demo.run, setup, runs=1)
    cvxpy_result = benchmark.run(cvxpy_microgrid_demo.solve_microgrid, setup, runs=1)
    pyomo_result = benchmark.run(pyomo_microgrid_demo.solve_microgrid_pyomo, setup, runs=1)
    gboml_result = benchmark.run(gboml_microgrid_demo.run, setup, runs=1)

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
        results[t] = measure_execution_time(setup=setup)

    # for n in no_homes:
    #     setup = MicrogridSetup()
    #     setup.no_homes = n
    #     setup.no_solar_panels = n
    #     setup.output_path = f"figures/output_homes{n}.txt"
    #     results[n] = measure_execution_time(setup=setup)

    plot_execution_time(results=results)
    # measure_lines_of_code(folders=["GBOML", "Pyomo", "CVXPY"])
