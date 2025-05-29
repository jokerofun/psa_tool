#!/usr/bin/env python3
"""
Simple script to test imports for the PSA tool
"""

import sys
import os

# Add project root to Python path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_root)

print(f"Using project root: {project_root}")

# Try imports
try:
    print("Trying src imports...")
    from src.dataflow.parallel_execution_dataflow import ParallelExecutionNode
    print("✓ Imported ParallelExecutionNode")
    
    from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
    print("✓ Imported ParallelDataFlowManager")
    
    from src.dataflow.dataflow_classes import DataProcessingNode
    print("✓ Imported DataProcessingNode")
    
    from src.optimization.solver_classes import Node
    print("✓ Imported Node")
    
except ImportError as e:
    print(f"Import error with src paths: {e}")
    print("Trying alternative import paths...")
    
    try:
        from dataflow.parallel_execution_dataflow import ParallelExecutionNode
        print("✓ Imported ParallelExecutionNode (alt)")
        
        from dataflow.dataflow_manager_paralell import ParallelDataFlowManager
        print("✓ Imported ParallelDataFlowManager (alt)")
        
        from dataflow.dataflow_classes import DataProcessingNode
        print("✓ Imported DataProcessingNode (alt)")
        
        from optimization.solver_classes import Node
        print("✓ Imported Node (alt)")
    except ImportError as e:
        print(f"Alternative import also failed: {e}")
        sys.exit(1)

# Test creating instances
print("\nTesting class instantiation:")

try:
    node = Node("test_node")
    print("✓ Created Node instance")
    
    dp_node = DataProcessingNode("test_dp_node", lambda x, y: None)
    print("✓ Created DataProcessingNode instance")
    
    parallel_node = ParallelExecutionNode("test_parallel_node")
    print("✓ Created ParallelExecutionNode instance")
    
    manager = ParallelDataFlowManager.getInstance()
    print("✓ Got ParallelDataFlowManager instance")
except Exception as e:
    print(f"Error creating instances: {e}")
    import traceback
    traceback.print_exc()
