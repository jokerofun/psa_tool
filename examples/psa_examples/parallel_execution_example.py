import time
import pandas as pd
import numpy as np
import sys
import os
import threading

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.optimization.solver_classes import Node

# Define a simple Node subclass for demonstration
class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

# Define various processing functions with different delays to demonstrate parallel execution

def processing_a(dfs, parameters):
    """Processing function with 2 second delay, no dependencies."""
    print(f"[{threading.current_thread().name}] Starting processing A with parameters: {parameters}")
    time.sleep(2)  # Sleep for 2 seconds
    
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': np.random.rand(10) * parameters.get('factor_a', 1.0)
    })
    
    print(f"[{threading.current_thread().name}] Processing A completed")
    return {'results': result_df}

def processing_b(dfs, parameters):
    """Processing function with 3 second delay, no dependencies."""
    print(f"[{threading.current_thread().name}] Starting processing B with parameters: {parameters}")
    time.sleep(3)  # Sleep for 3 seconds
    
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': np.random.rand(10) * parameters.get('factor_b', 1.0)
    })
    
    print(f"[{threading.current_thread().name}] Processing B completed")
    return {'results': result_df}

def processing_c(dfs, parameters):
    """Processing function with 1 second delay, depends on A and B."""
    print(f"[{threading.current_thread().name}] Starting processing C with parameters: {parameters}")
    # Use input data from dependencies
    print(f"[{threading.current_thread().name}] C received {len(dfs)} input dataframes")
    
    time.sleep(1)  # Sleep for 1 second
    
    # Combine inputs if available
    combined_values = np.random.rand(10)  # Default values
    if dfs:
        for key, df in dfs.items():
            if 'values' in df.columns:
                combined_values = combined_values + df['values'].values
    
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': combined_values * parameters.get('factor_c', 1.0)
    })
    
    print(f"[{threading.current_thread().name}] Processing C completed")
    return {'results': result_df}

def processing_d(dfs, parameters):
    """Processing function with 2 second delay, depends on B."""
    print(f"[{threading.current_thread().name}] Starting processing D with parameters: {parameters}")
    time.sleep(2)  # Sleep for 2 seconds
    
    # Use B's output if available
    b_values = np.random.rand(10)  # Default values
    if dfs:
        for key, df in dfs.items():
            if 'values' in df.columns:
                b_values = df['values'].values
    
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': b_values * 2 * parameters.get('factor_d', 1.0)
    })
    
    print(f"[{threading.current_thread().name}] Processing D completed")
    return {'results': result_df}

def processing_final(dfs, parameters):
    """Final processing function, depends on C and D."""
    print(f"[{threading.current_thread().name}] Starting FINAL processing with parameters: {parameters}")
    time.sleep(1)  # Sleep for 1 second
    
    # Combine all inputs
    combined_values = np.zeros(10)
    if dfs:
        for key, df in dfs.items():
            if 'values' in df.columns:
                combined_values = combined_values + df['values'].values
    
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': combined_values * parameters.get('factor_final', 1.0)
    })
    
    print(f"[{threading.current_thread().name}] FINAL processing completed")
    return {'results': result_df}

def main():
    # Get the parallel execution dataflow manager instance
    manager = ParallelExecutionDataFlowManager.getInstance()
    
    # Create a dataflow
    dataflow = manager.newDataFlow(DemoNode)
    
    # Set up the dataflow with nodes and dependencies
    print("Setting up dataflow nodes...")
    
    # Create a more complex dataflow with more parallel paths to better demonstrate speedup
    
    # Independent nodes (level 1) - these will execute in parallel
    node_a1 = dataflow.node("A1", ParallelExecutionNode)
    node_a1.process_func = processing_a
    
    node_a2 = dataflow.node("A2", ParallelExecutionNode)
    node_a2.process_func = processing_a  # Reuse the same function
    
    node_b1 = dataflow.node("B1", ParallelExecutionNode)
    node_b1.process_func = processing_b
    
    node_b2 = dataflow.node("B2", ParallelExecutionNode)
    node_b2.process_func = processing_b  # Reuse the same function
    
    # Level 2 nodes (depend on level 1) - these can execute once their dependencies complete
    node_c1 = dataflow.node("C1", ParallelExecutionNode)
    node_c1.process_func = processing_c
    node_c1.add_dependency(node_a1)
    node_c1.add_dependency(node_b1)
    
    node_c2 = dataflow.node("C2", ParallelExecutionNode)
    node_c2.process_func = processing_c
    node_c2.add_dependency(node_a2)
    node_c2.add_dependency(node_b2)
    
    node_d1 = dataflow.node("D1", ParallelExecutionNode)
    node_d1.process_func = processing_d
    node_d1.add_dependency(node_b1)
    
    node_d2 = dataflow.node("D2", ParallelExecutionNode)
    node_d2.process_func = processing_d
    node_d2.add_dependency(node_b2)
    
    # Final node (level 3)
    node_final = dataflow.node("Final", ParallelExecutionNode, final=True)
    node_final.process_func = processing_final
    node_final.add_dependency(node_c1)
    node_final.add_dependency(node_c2)
    node_final.add_dependency(node_d1)
    node_final.add_dependency(node_d2)
    
    # Can also use the >> operator for dependencies
    # node_a >> node_c
    # node_b >> node_c
    # node_b >> node_d
    # node_c >> node_final
    # node_d >> node_final
    
    print("Dataflow setup complete. Executing...")
    
    # Start timing
    start_time = time.time()
    
    # Execute the dataflow
    results = manager.executeDataFlow(DemoNode, {
        'factor_a': 2.0,
        'factor_b': 3.0,
        'factor_c': 1.5,
        'factor_d': 2.5,
        'factor_final': 0.5
    })
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nExecution completed in {elapsed_time:.2f} seconds")
    
    # Print results
    for node_name, result in results.items():
        print(f"Results from {node_name}: {result[:3]}...")
    
    print("\nSequential execution would have taken approximately 18 seconds (2x2 + 2x3 + 2x1 + 2x2 + 1)")
    print(f"Parallel execution took {elapsed_time:.2f} seconds")
    print("Speedup ratio: {:.2f}x".format(18 / elapsed_time))
    
    # Shutdown the executor
    manager.shutdown()

if __name__ == "__main__":
    main()
