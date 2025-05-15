import time
from typing import Any, Callable, Dict


class Benchmark:
    @staticmethod
    def run(func: Callable, *args, runs: int = 1, **kwargs) -> Dict[str, Any]:
        """
       Benchmarks the given function by executing it multiple times.

       Parameters:
           func (Callable): The function to benchmark.
           *args: Positional arguments to pass to the function.
           runs (int): Number of times to run the function (default is 1).
           **kwargs: Keyword arguments to pass to the function.

       Returns:
           dict: A dictionary with benchmark statistics and the last result.
       """

        times = []
        result = None

        print(f"Benchmarking '{func.__name__}' for {runs} run(s)...")

        for i in range(1, runs + 1):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end - start
            times.append(duration)
            print(f"Run {i}: {duration:.4f} seconds")

        stats = {
            'function': func.__name__,
            'runs': runs,
            'total_time': sum(times),
            'average_time': sum(times) / runs,
            'min_time': min(times),
            'max_time': max(times),
            'last_result': result
        }

        print("Benchmark Summary:")
        print(f"Total Time: {stats['total_time']:.6f} seconds")
        print(f"Average Time: {stats['average_time']:.6f} seconds")
        print(f"Min Time: {stats['min_time']:.6f} seconds")
        print(f"Max Time: {stats['max_time']:.6f} seconds")
        print(f"Last Result: {stats['last_result']}")

        return stats


if __name__ == "__main__":
    # Example usage
    def example_function(x, y):
        time.sleep(0.1)  # Simulate some processing time
        return x + y

    benchmark = Benchmark()
    benchmark.run(example_function, 5, 10, runs=3)
