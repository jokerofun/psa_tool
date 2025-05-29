#!/usr/bin/env python3
"""
Simple test of parallel dataflow
"""

import time
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import directly like the working example
try:
    from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
    from src.optimization.solver_classes import Node
    print("Successfully imported from src paths")
except ImportError:
    try:
        from dataflow.dataflow_manager_paralell import ParallelDataFlowManager
        from optimization.solver_classes import Node
        print("Successfully imported from direct paths")
    except ImportError:
        print("Both import paths failed")
        sys.exit(1)

# Define a simple Node subclass for demonstration
class TestNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

# Define a simple processing function
def simple_process(dfs, parameters):
    """A simple processing function that returns a dictionary with a DataFrame"""
    print(f"Processing with parameters: {parameters}")
    
    # Create a simple DataFrame
    df = pd.DataFrame({
        'value': [i * parameters.get('multiplier', 1) for i in range(10)]
    })
    
    print(f"Returning DataFrame with shape {df.shape}")
    
    # Return a dictionary containing the DataFrame
    return {'result': df}

def main():
    print("Starting simple parallel dataflow test")
    
    # Get the manager instance
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        print("Creating dataflow")
        dataflow = manager.newDataFlow(TestNode)
        
        # Add a node to the dataflow
        print("Adding node to dataflow")
        node = dataflow.node("test_node", None, final=True)
        node.process_func = simple_process
        
        # Execute the dataflow
        print("Executing dataflow")
        task_id = manager.executeDataFlow(TestNode, {'multiplier': 2})
        
        print(f"Task ID: {task_id}")
        
        # Wait for the result
        print("Waiting for result...")
        result = manager.getData(task_id, wait=True)
        
        print(f"Got result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown the manager
        print("Shutting down manager")
        manager.shutdown()
    
    print("Test completed")
    
if __name__ == "__main__":
    main()
