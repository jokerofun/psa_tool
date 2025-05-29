#!/usr/bin/env python3
"""
Versioned DuckDB Execution: A DuckDB parallel execution implementation with versioning.

This module extends DuckDB parallel execution with versioning capabilities to:
1. Cache results based on input data, parameters, and processing functions
2. Avoid recomputation by reusing cached results when inputs are the same
3. Track result lineage and versions
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Callable
import threading
import pandas as pd
import time
import uuid

from ..duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from ..duckdb_dataflow_utils import DuckDBDataflowManager
from ..parallel_execution_dataflow import ParallelExecutionDataFlowManager

from .version_utils import generate_version_key, is_result_cacheable
from .version_cache import get_version_cache

class VersionedDuckDBExecutionNode(DuckDBParallelExecutionNode):
    """
    A DuckDB parallel execution node with versioning capabilities.
    
    This node extends DuckDBParallelExecutionNode to add:
    1. Result caching based on input data, parameters, and processing function
    2. Version tracking for results
    3. Ability to skip computation by reusing cached results
    """
    
    def __init__(self, name: str, final=False):
        """
        Initialize a versioned DuckDB parallel execution node.
        
        Args:
            name: The name of the node
            final: Whether this node is final
        """
        super().__init__(name, final)
        self._version_cache = get_version_cache()
        self._current_version_key = None
        self._cache_enabled = True  # Can be configured per node
    
    def set_cache_enabled(self, enabled: bool):
        """
        Enable or disable result caching for this node.
        
        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
    
    def is_cache_enabled(self) -> bool:
        """
        Check if caching is enabled for this node.
        
        Returns:
            bool: True if caching is enabled
        """
        return self._cache_enabled
    
    def _should_use_cache(self, func: Callable) -> bool:
        """
        Determine if caching should be used for this execution.
        
        Args:
            func: The processing function
        
        Returns:
            bool: True if caching should be used
        """
        # Check node-level setting
        if not self._cache_enabled:
            return False
        
        # Check function-level setting (from decorators or similar)
        return is_result_cacheable(func)
    
    def parallel_run(self, parameters={}):
        """
        Execute this node with versioning support.
        
        If a previous version with identical inputs exists, reuse it.
        Otherwise, compute and cache the result.
        
        Args:
            parameters: Execution parameters
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
            
            print(f"{self.name} is running in versioned execution with {len(input_dfs)} input dataframes")
            
            # Generate a version key for this execution
            version_key = None
            if self.process_func and self._should_use_cache(self.process_func):
                version_key = generate_version_key(self.process_func, input_dfs, parameters)
                self._current_version_key = version_key
                
                # Check if we have a cached version
                if self._version_cache.version_exists(version_key):
                    print(f"Cache hit for node {self.name} - retrieving cached results")
                    
                    # Get results from cache
                    cached_results = self._version_cache.get_results(version_key)
                    
                    if cached_results:
                        # Store in DuckDB for this execution
                        self._result_ids = self._db_manager.store_node_results(
                            self._execution_id, self.name, cached_results)
                        self._results = cached_results
                        self._has_stored_results = True
                        
                        print(f"Node {self.name} using cached results with {len(cached_results)} dataframes")
                        
                        # Mark execution as finished and return early
                        self.mark_execution_finished()
                        return
            
            # No cache hit or caching disabled - process normally
            print(f"No cache hit for node {self.name} - computing results")
            
            # Process the inputs
            result_dfs = self.process(input_dfs, parameters)
            
            # Store the results in DuckDB
            if result_dfs is not None:
                print(f"Node {self.name} processing completed successfully with {len(result_dfs)} result dataframes")
                self._result_ids = self._db_manager.store_node_results(
                    self._execution_id, self.name, result_dfs)
                self._results = result_dfs
                self._has_stored_results = True
                
                # Cache the results if version_key was generated
                if version_key:
                    # Create input signature for better tracking
                    input_signature = {
                        'parameter_keys': list(parameters.keys()),
                        'input_dataframes': {name: {'shape': [len(df), len(df.columns)], 
                                                  'columns': list(df.columns)} 
                                           for name, df in input_dfs.items()}
                    }
                    
                    print(f"Caching results for node {self.name} with version key {version_key[:8]}...")
                    self._version_cache.store_results(
                        version_key, self.name, result_dfs,
                        input_signature=input_signature
                    )
                
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
    
    def get_current_version_key(self) -> Optional[str]:
        """
        Get the version key of the most recent execution.
        
        Returns:
            Optional[str]: The version key or None if unavailable
        """
        return self._current_version_key


class VersionedDuckDBDataflow(DuckDBParallelDataflow):
    """
    A DuckDB parallel dataflow with versioning capabilities.
    
    This class extends DuckDBParallelDataflow to add:
    1. Version tracking for the entire dataflow
    2. Ability to use versioned nodes
    """
    
    def __init__(self, base_dataflow):
        """
        Initialize a versioned DuckDB dataflow.
        
        Args:
            base_dataflow: The ParallelExecutionDataflow instance to wrap
        """
        super().__init__(base_dataflow)
        self._version_cache = get_version_cache()
        self._versioning_enabled = True
    
    def set_versioning_enabled(self, enabled: bool):
        """
        Enable or disable versioning for this dataflow.
        
        Args:
            enabled: Whether to enable versioning
        """
        self._versioning_enabled = enabled
        
        # Also set for all versioned nodes
        for _, node in self.base_dataflow.nodes.items():
            if isinstance(node, VersionedDuckDBExecutionNode):
                node.set_cache_enabled(enabled)
    
    def is_versioning_enabled(self) -> bool:
        """
        Check if versioning is enabled for this dataflow.
        
        Returns:
            bool: True if versioning is enabled
        """
        return self._versioning_enabled
    
    def execute(self, parameters={}) -> Dict[str, Any]:
        """
        Execute the dataflow with versioning support.
        
        Args:
            parameters: Parameters for this execution
        
        Returns:
            Dict[str, Any]: The results of the execution
        """
        # Use the base implementation
        results = super().execute(parameters)
        
        # Collect version information for reporting
        if self._versioning_enabled:
            version_info = {}
            for node_name, node in self.base_dataflow.nodes.items():
                if isinstance(node, VersionedDuckDBExecutionNode):
                    version_key = node.get_current_version_key()
                    if version_key:
                        version_info[node_name] = version_key
            
            print(f"Dataflow executed with {len(version_info)} versioned nodes")
        
        return results

# Factory function for creating versioned nodes and dataflows
def create_versioned_dataflow(node_class) -> VersionedDuckDBDataflow:
    """
    Create a versioned DuckDB dataflow for the given node class.
    
    Args:
        node_class: The node class to use
    
    Returns:
        VersionedDuckDBDataflow: A versioned dataflow
    """
    manager = ParallelExecutionDataFlowManager.getInstance()
    dataflow = manager.newDataFlow(node_class)
    return VersionedDuckDBDataflow(dataflow)
