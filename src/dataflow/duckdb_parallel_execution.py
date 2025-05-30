from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import threading
import pandas as pd
import time
import uuid

from .parallel_execution_dataflow import ParallelExecutionNode
from .duckdb_dataflow_utils import DuckDBDataflowManager

class DuckDBParallelExecutionNode(ParallelExecutionNode):
    """
    A parallel execution node that uses DuckDB for data persistence.
    This allows for efficient handling of large dataframes and enables
    dataflow to survive application restarts.
    """
    
    def __init__(self, name: str, final=False):
        """
        Initialize a new DuckDB-backed parallel execution node.
        
        Args:
            name: The name of the node
            final: Whether this node is a final node in the dataflow
        """
        super().__init__(name, final)
        self._db_manager = DuckDBDataflowManager.get_instance()
        self._execution_id = None
        self._result_ids = {}
        self._has_stored_results = False
    
    def set_execution_id(self, execution_id: str):
        """
        Set the execution ID for this node.
        
        Args:
            execution_id: The execution ID to use for database operations
        """
        self._execution_id = execution_id
    
    def parallel_run(self, parameters={}):
        """
        Execute this node's processing logic with DuckDB persistence.
        
        Args:
            parameters: Parameters for this execution
        """
        # Ensure we have an execution ID
        if not self._execution_id:
            raise ValueError("No execution ID set for node. Call set_execution_id() before running.")
        
        # Mark execution as started
        self.mark_execution_started()
        
        try:
            # Get dependency node names
            dep_node_names = [node.name for node in self._dependencies]
            
            # Collect input dataframes from dependencies using DuckDB
            input_dfs = self._db_manager.get_dependency_results(self._execution_id, dep_node_names)
            
            print(f"{self.name} is running in parallel execution with {len(input_dfs)} input dataframes from DuckDB")
            
            # Process the inputs
            result_dfs = self.process(input_dfs, parameters)
            
            # Store the results in DuckDB
            if result_dfs is not None:
                print(f"Node {self.name} processing completed successfully with {len(result_dfs)} result dataframes")
                self._result_ids = self._db_manager.store_node_results(
                    self._execution_id, self.name, result_dfs)
                self._results = result_dfs  # Also keep in memory for direct access
                self._has_stored_results = True
            elif not self._has_stored_results and input_dfs:
                print(f"Node {self.name} returned no results, using input dataframes")
                self._result_ids = self._db_manager.store_node_results(
                    self._execution_id, self.name, input_dfs)
                self._results = input_dfs
                self._has_stored_results = True
            else:
                print(f"Node {self.name} has no results and no input dataframes")
        
        except Exception as e:
            print(f"Error executing {self.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Mark execution as finished
            self.mark_execution_finished()
    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        """
        Get the results of this node's execution.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of result dataframes
        """
        # If results are in memory, return them
        if self._results and len(self._results) > 0:
            return self._results
        
        # If we've stored results in DuckDB and have an execution ID, load them
        if self._has_stored_results and self._execution_id:
            results = self._db_manager.load_node_results(self._execution_id, self.name)
            self._results = results
            return results
        
        return {}
    
    def clear_results(self):
        """Clear the results from memory (but not from DuckDB)."""
        self._results = {}


class DuckDBParallelDataflow:
    """
    A dataflow that uses DuckDB for data persistence.
    This class wraps ParallelExecutionDataflow to provide DuckDB integration.
    """
    
    def __init__(self, base_dataflow):
        """
        Initialize a new DuckDB-backed dataflow.
        
        Args:
            base_dataflow: The ParallelExecutionDataflow instance to wrap
        """
        self.base_dataflow = base_dataflow
        self._db_manager = DuckDBDataflowManager.get_instance()
        self._execution_id = None
        self.NodeClass = getattr(base_dataflow, 'NodeClass', None)
    
    def execute(self, parameters={}) -> Dict[str, Any]:
        """
        Execute the dataflow with DuckDB persistence.
        
        Args:
            parameters: Parameters for this execution
        
        Returns:
            Dict[str, Any]: The results of the execution
        """
        # Start a new execution
        dataflow_name = self.NodeClass.__name__ if self.NodeClass else "UnknownDataflow"
        self._execution_id = self._db_manager.start_execution(dataflow_name, parameters)
        
        # Set execution ID for all nodes
        for node_name, node in self.base_dataflow.nodes.items():
            if isinstance(node, DuckDBParallelExecutionNode):
                node.set_execution_id(self._execution_id)
        
        try:
            # Execute the dataflow
            results = self.base_dataflow.execute(parameters)
            self._db_manager.complete_execution(self._execution_id, 'completed')
            return results
        except Exception as e:
            self._db_manager.complete_execution(self._execution_id, 'error')
            raise
    
    def clear_execution_data(self):
        """Clear all data associated with the current execution."""
        if self._execution_id:
            self._db_manager.clear_execution_data(self._execution_id)
