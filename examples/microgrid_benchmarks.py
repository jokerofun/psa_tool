
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.benchmark import Benchmark
from psa_examples.microgrid_setup import MicrogridSetup
import examples.psa_examples.native_cvxpy_microgrid_solution as cvxpy_microgrid_demo
import examples.psa_examples.native_pyomo_microgrid_solution as pyomo_microgrid_demo
import examples.psa_examples.gboml_microgrid_solution as gboml_microgrid_demo
import microgrid_demo
from examples.helpers.file_writer import empty

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
    plt.savefig("figure/execution_time_comparison.png")
    plt.show()



if __name__ == "__main__":
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
