import time
import pandas as pd
import numpy as np
import sys
import os
import threading

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
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

# Define processing functions
def processing_a(dfs, parameters):
    """Processing function with 2 second delay, generates a large dataframe."""
    print(f"[{threading.current_thread().name}] Starting processing A with parameters: {parameters}")
    time.sleep(2)  # Sleep for 2 seconds
    
    # Create a large dataframe (but not too large to cause overflow)
    size = min(parameters.get('size', 100_000), 10_000)  # Limit to 10,000 rows
    
    # Generate timestamps day by day to avoid overflow
    timestamps = [pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(size)]
    
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'values': np.random.rand(size) * parameters.get('factor_a', 1.0),
        'category': np.random.choice(['A', 'B', 'C'], size=size)
    })
    
    print(f"[{threading.current_thread().name}] Processing A completed with {len(result_df)} rows")
    return {'results': result_df}

def processing_b(dfs, parameters):
    """Processing function with 1 second delay, depends on A."""
    print(f"[{threading.current_thread().name}] Starting processing B with parameters: {parameters}")
    
    # Get input dataframe
    input_df = None
    for key, df in dfs.items():
        input_df = df
        break
        
    if input_df is None:
        print("No input dataframe found!")
        return {'results': pd.DataFrame()}
        
    time.sleep(1)  # Sleep for 1 second
    
    # Filter the dataframe based on category
    result_df = input_df[input_df['category'] == 'A'].copy()
    result_df['values'] = result_df['values'] * parameters.get('factor_b', 2.0)
    
    print(f"[{threading.current_thread().name}] Processing B completed with {len(result_df)} rows")
    return {'filtered_results': result_df}

def processing_c(dfs, parameters):
    """Processing function with 1 second delay, depends on B."""
    print(f"[{threading.current_thread().name}] Starting processing C with parameters: {parameters}")
    
    # Get input dataframe
    input_df = None
    for key, df in dfs.items():
        input_df = df
        break
        
    if input_df is None:
        print("No input dataframe found!")
        return {'results': pd.DataFrame()}
    
    time.sleep(1)  # Sleep for 1 second
    
    # Aggregate the dataframe
    result_df = input_df.resample('D', on='timestamp').agg({'values': 'sum'}).reset_index()
    
    print(f"[{threading.current_thread().name}] Processing C completed with {len(result_df)} rows")
    return {'aggregated_results': result_df}

def main():
    # Get the parallel execution dataflow manager instance
    manager = ParallelExecutionDataFlowManager.getInstance()
    
    # Create a dataflow
    dataflow = manager.newDataFlow(DemoNode)
    
    # Make sure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'duckdb')
    os.makedirs(data_dir, exist_ok=True)
    
    # Wrap the dataflow with DuckDB persistence
    duckdb_dataflow = DuckDBParallelDataflow(dataflow)
    
    # Set up the dataflow with nodes and dependencies
    print("Setting up dataflow nodes with DuckDB persistence...")
    
    # Create nodes with DuckDB persistence
    node_a = dataflow.node("A", DuckDBParallelExecutionNode)
    node_a.process_func = processing_a
    print(f"Created node A of type {type(node_a)}")
    
    node_b = dataflow.node("B", DuckDBParallelExecutionNode)
    node_b.process_func = processing_b
    node_b.add_dependency(node_a)
    print(f"Created node B of type {type(node_b)}")
    
    node_c = dataflow.node("C", DuckDBParallelExecutionNode, final=True)
    node_c.process_func = processing_c
    node_c.add_dependency(node_b)
    print(f"Created node C of type {type(node_c)}")
    
    print("Dataflow setup complete. Executing...")
    
    # Measure memory usage before execution
    import psutil
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Start timing
    start_time = time.time()
    
    # Execute the dataflow with large dataframes
    results = duckdb_dataflow.execute({
        'size': 1_000_000,  # 1 million rows
        'factor_a': 2.0,
        'factor_b': 3.0,
    })
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Measure memory after execution
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\nExecution completed in {elapsed_time:.2f} seconds")
    print(f"Memory usage: {memory_before:.2f} MB before, {memory_after:.2f} MB after")
    
    # Print results - should be aggregated data from node C
    print("\nFinal Results:")
    for node_name, result in results.items():
        if isinstance(result, (list, np.ndarray)):
            print(f"Results from {node_name}: {result[:5]}")
    
    # Load results from DuckDB as a demonstration
    from src.dataflow.duckdb_dataflow_utils import DuckDBDataflowManager
    db_manager = DuckDBDataflowManager.get_instance()
    
    # Get all node results from DuckDB
    print("\nRetrieving results directly from DuckDB:")
    # Peek into the execution table to see what's there
    execution_info = db_manager.conn.execute("SELECT * FROM dataflow_executions").fetchall()
    print(f"Found {len(execution_info)} execution records in DuckDB")
    
    if execution_info:
        print(f"Latest execution ID: {execution_info[-1][0]}")
        # Use the known execution ID from our wrapped dataflow
        exec_id = duckdb_dataflow._execution_id
        
        # List all stored results for this execution
        results_info = db_manager.conn.execute(
            "SELECT node_name, table_name FROM node_results WHERE execution_id = ?", 
            (exec_id,)
        ).fetchall()
        print(f"Found {len(results_info)} result records for execution {exec_id}")
        
        # Try to load results for node C
        node_c_results = db_manager.load_node_results(exec_id, "C")
        if node_c_results:
            for result_name, df in node_c_results.items():
                print(f"Node C '{result_name}' from DuckDB ({len(df)} rows):")
                print(df.head(5))
    
    # Shutdown the executor
    manager.shutdown()
    
    print("\nDuckDB dataflow execution complete!")
    print("Results are persisted in the DuckDB database even after the program exits.")
    print("You can query them directly from the database or through the DuckDBDataflowManager API.")

if __name__ == "__main__":
    main()
