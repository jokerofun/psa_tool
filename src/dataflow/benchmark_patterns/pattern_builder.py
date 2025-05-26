"""
Benchmark pattern utilities for dataflow performance testing.
This module provides setup helpers for creating more complex benchmark scenarios.
"""

from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager, ParallelExecutionNode
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow
from src.optimization.solver_classes import Node

class BenchmarkPatternBuilder:
    """
    Helper class to set up benchmark patterns using either in-memory or DuckDB nodes.
    """
    
    def __init__(self, use_duckdb=False):
        """
        Initialize the pattern builder.
        
        Args:
            use_duckdb: Whether to use DuckDB nodes (True) or in-memory nodes (False)
        """
        self.use_duckdb = use_duckdb
        self.manager = ParallelExecutionDataFlowManager.getInstance()
        self.dataflow = self.manager.newDataFlow(Node)
        
        # If using DuckDB, wrap the dataflow
        if use_duckdb:
            self.wrapped_dataflow = DuckDBParallelDataflow(self.dataflow)
        else:
            self.wrapped_dataflow = self.dataflow
    
    def create_node(self, name, process_func, dependencies=None, final=False):
        """
        Create a new node in the dataflow.
        
        Args:
            name: Name of the node
            process_func: Function to process data
            dependencies: List of nodes this node depends on
            final: Whether this is a final node
            
        Returns:
            The created node
        """
        # Create appropriate node type
        if self.use_duckdb:
            node = self.dataflow.node(name, DuckDBParallelExecutionNode, final=final)
        else:
            node = self.dataflow.node(name, ParallelExecutionNode, final=final)
        
        # Set process function
        node.process_func = process_func
        
        # Add dependencies if provided
        if dependencies:
            for dep_node in dependencies:
                node.add_dependency(dep_node)
        
        return node
    
    def build_linear_pattern(self, processing_funcs):
        """
        Build a linear pattern where each node depends on the previous one.
        
        Args:
            processing_funcs: Dictionary of processing functions
            
        Returns:
            The wrapped dataflow
        """
        # Create nodes in sequence
        node_a = self.create_node("A", processing_funcs['processing_a'])
        node_b = self.create_node("B", processing_funcs['processing_b'], [node_a])
        node_c = self.create_node("C", processing_funcs['processing_c'], [node_b])
        node_d = self.create_node("D", processing_funcs['processing_d'], [node_c], final=True)
        
        return self.wrapped_dataflow
    
    def build_star_pattern(self, processing_funcs):
        """
        Build a star pattern with multiple source nodes feeding into a central node.
        
        Args:
            processing_funcs: Dictionary of processing functions
            
        Returns:
            The wrapped dataflow
        """
        # Create source nodes
        node_source1 = self.create_node("Source1", processing_funcs['processing_source1'])
        node_source2 = self.create_node("Source2", processing_funcs['processing_source2'])
        node_source3 = self.create_node("Source3", processing_funcs['processing_source3'])
        
        # Create central node with all sources as dependencies
        node_central = self.create_node("Central", processing_funcs['processing_central'], 
                                     [node_source1, node_source2, node_source3], final=True)
        
        return self.wrapped_dataflow
    
    def build_tree_pattern(self, processing_funcs):
        """
        Build a tree pattern with a root node, branch nodes, and leaf nodes.
        
        Args:
            processing_funcs: Dictionary of processing functions
            
        Returns:
            The wrapped dataflow
        """
        # Create root node
        node_root = self.create_node("Root", processing_funcs['processing_root'])
        
        # Create branch nodes
        node_branch1 = self.create_node("Branch1", processing_funcs['processing_branch1'], [node_root])
        node_branch2 = self.create_node("Branch2", processing_funcs['processing_branch2'], [node_root])
        node_branch3 = self.create_node("Branch3", processing_funcs['processing_branch3'], [node_root])
        
        # Create leaf nodes
        node_leaf1 = self.create_node("Leaf1", processing_funcs['processing_leaf1'], [node_branch1], final=True)
        node_leaf2 = self.create_node("Leaf2", processing_funcs['processing_leaf2'], 
                                   [node_branch2, node_branch3], final=True)
        
        return self.wrapped_dataflow
    
    def build_cyclic_pattern(self, processing_funcs):
        """
        Build a pattern that simulates iterative processing.
        
        Args:
            processing_funcs: Dictionary of processing functions
            
        Returns:
            The wrapped dataflow
        """
        # Note: This doesn't actually create cycles in the dataflow graph
        # since that's not allowed, but it simulates iterative processing
        # through custom logic in the processing functions
        
        # Create nodes
        node_start = self.create_node("Start", processing_funcs['processing_start'])
        node_iterate = self.create_node("Iterate", processing_funcs['processing_iterate'], [node_start])
        node_checkpoint = self.create_node("Checkpoint", processing_funcs['processing_checkpoint'], [node_iterate])
        node_final = self.create_node("Final", processing_funcs['processing_final'], [node_iterate], final=True)
        
        return self.wrapped_dataflow
