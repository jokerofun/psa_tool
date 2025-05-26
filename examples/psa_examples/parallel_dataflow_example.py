import time
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
from src.dataflow.parallel_dataflow_classes import ParallelProcessingNode
from src.dataflow.dataflow_classes import DataProcessingNode
from src.optimization.solver_classes import Node
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
from src.dataflow.parallel_dataflow_classes import ParallelProcessingNode
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

# Define some processing functions for demonstration
def slow_processing(dfs, parameters):
    """A function that simulates a slow processing operation."""
    print(f"Starting slow process with parameters: {parameters}")
    # Simulate a CPU-intensive task
    time.sleep(5)  # Sleep for 5 seconds to simulate work
    
    # Create a simple DataFrame with some results
    # Use the required column name that matches _final_df_name in ParallelDataflow
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': np.random.rand(10) * parameters.get('factor', 1.0)
    })
    
    print("Slow process completed")
    # Return with the key that matches _final_df_name in ParallelDataflow
    return {'results': result_df}

def quick_processing(dfs, parameters):
    """A function that simulates a quick processing operation."""
    print(f"Starting quick process with parameters: {parameters}")
    # Simulate a quick task
    time.sleep(1)  # Sleep for 1 second to simulate work
    
    # Create a simple DataFrame with some results
    # Use the required column name that matches _final_df_name in ParallelDataflow
    result_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'values': np.random.rand(10) * parameters.get('factor', 1.0)
    })
    
    print("Quick process completed")
    # Return with the key that matches _final_df_name in ParallelDataflow
    return {'results': result_df}

def main():
    # Get the parallel dataflow manager instance
    manager = ParallelDataFlowManager.getInstance()
    
    # Create two dataflows for different scenarios
    slow_dataflow = manager.newDataFlow(DemoNode)
    quick_dataflow = manager.newDataFlow(DemoNode)
    
    # Set up the slow dataflow
    slow_node = slow_dataflow.node("slow_processor", ParallelProcessingNode, final=True)
    slow_node.process_func = slow_processing
    
    # Set up the quick dataflow
    quick_node = quick_dataflow.node("quick_processor", ParallelProcessingNode, final=True)
    quick_node.process_func = quick_processing
    
    # Execute both dataflows in parallel with different parameters
    slow_task_id = manager.executeDataFlow(DemoNode, {'factor': 10.0})
    quick_task_id = manager.executeDataFlow(DemoNode, {'factor': 5.0})
    
    print(f"Submitted tasks: slow_task_id={slow_task_id}, quick_task_id={quick_task_id}")
    
    # Print status of both tasks
    print("\nInitial Status:")
    print(f"Slow Task Status: {manager.getTaskStatus(slow_task_id)}")
    print(f"Quick Task Status: {manager.getTaskStatus(quick_task_id)}")
    
    # Wait for the quick task to complete and get its results
    print("\nWaiting for quick task to complete...")
    quick_result = manager.getData(quick_task_id, wait=True)
    # check if it has 3 values
    if quick_result is not None:
        print(f"Quick task result: {quick_result[:3]}...")  # Show first 3 values
    else: 
        print("Quick task result is None, something went wrong.")
    # Check if the slow task is done yet (it shouldn't be)
    print("\nChecking if slow task is completed:")
    is_slow_completed = manager.isCompleted(slow_task_id)
    print(f"Is slow task completed? {is_slow_completed}")
    
    # Wait for the slow task to complete and get its results
    print("\nWaiting for slow task to complete...")
    slow_result = manager.getData(slow_task_id, wait=True)
    if slow_result is not None:
        print(f"Slow task result: {slow_result[:3]}...")  # Show first 3 values
    else:  
        print("Slow task result is None, something went wrong.")
    # Check final status
    print("\nFinal Status:")
    print(f"Slow Task Status: {manager.getTaskStatus(slow_task_id)}")
    print(f"Quick Task Status: {manager.getTaskStatus(quick_task_id)}")
    
    # Shutdown the executor (this is important to prevent the program from hanging)
    manager.shutdown()
    
if __name__ == "__main__":
    main()
