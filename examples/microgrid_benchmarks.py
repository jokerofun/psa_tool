
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.benchmark import Benchmark
from psa_examples.microgrid_setup import MicrogridSetup
import examples.psa_examples.native_cvxpy_microgrid_solution as cvxpy_microgrid_demo
import examples.psa_examples.gboml_microgrid_solution as gboml_microgrid_demo
import microgrid_demo

if __name__ == "__main__":
    setup = MicrogridSetup()

    benchmark = Benchmark()
    benchmark.run(gboml_microgrid_demo.run, setup, runs=1)
