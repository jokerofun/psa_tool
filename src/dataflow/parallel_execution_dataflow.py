from __future__ import annotations
import concurrent.futures
from typing import Dict, Any, List, Set, Optional
import threading
import time
import networkx as nx  # For dependency graph management
import pandas as pd
from collections import deque

from .parallel_dataflow_classes import ParallelDataflowNode, ParallelProcessingNode
from .dataflow_classes import DataflowNode
from .dataflow import Dataflow
from src.optimization.solver_classes import Node

class ParallelExecutionNode(ParallelDataflowNode):
    """
    A node class specifically designed for parallel execution within a dataflow.
    This class extends ParallelDataflowNode with additional properties to track
    execution status and manage dependencies.
    """
    def __init__(self, name: str, final=False):
        super().__init__(name, final)
        self._execution_started = False
        self._execution_finished = False
        self._execution_lock = threading.Lock()
        self._result_ready = threading.Event()
        self.process_func = None  # Will be set later
    
    def process(self, dfs: Dict[str, pd.DataFrame], parameters={}) -> Dict[str, pd.DataFrame]:
        """
        Override the process method to use the assigned process_func if available.
        If process_func is not set, simply return the input dataframes as the result.
        """
        if self.process_func:
            return self.process_func(dfs, parameters)
        # If no process_func is set, just pass through the input data
        print(f"Warning: Node {self.name} has no process_func defined, returning input dataframes")
        return dfs
    
    def mark_execution_started(self):
        """Mark this node as currently executing."""
        with self._execution_lock:
            self._execution_started = True
    
    def mark_execution_finished(self):
        """Mark this node as finished executing."""
        with self._execution_lock:
            self._execution_finished = True
            self._result_ready.set()
    
    def is_execution_started(self) -> bool:
        """Check if this node has started executing."""
        with self._execution_lock:
            return self._execution_started
    
    def is_execution_finished(self) -> bool:
        """Check if this node has finished executing."""
        with self._execution_lock:
            return self._execution_finished
    
    def wait_for_result(self, timeout=None) -> bool:
        """Wait for this node to finish execution."""
        return self._result_ready.wait(timeout)
    
    def parallel_run(self, parameters={}):
        """
        Execute this node's processing logic without recursively running dependencies.
        This allows the parallel executor to manage execution order.
        """
        # Mark execution as started
        self.mark_execution_started()
        
        try:
            # Collect input dataframes from dependencies
            # Dependencies are assumed to be already executed
            input_dfs = {}
            for node in self._dependencies:
                # Get results from each dependency
                dep_results = node.get_results()
                if dep_results:
                    input_dfs.update(dep_results)
            
            print(f"{self.name} is running in parallel execution with {len(input_dfs)} input dataframes")
            
            # Verify we have a process function
            if not hasattr(self, 'process_func') or self.process_func is None:
                print(f"Warning: Node {self.name} has no process_func defined")
            
            # Process the inputs
            result_dfs = self.process(input_dfs, parameters)
            
            # Store the results
            if result_dfs is not None:
                print(f"Node {self.name} processing completed successfully with {len(result_dfs)} result dataframes")
                self._results = result_dfs
            elif not self._results and input_dfs:
                print(f"Node {self.name} returned no results, using input dataframes")
                self._results = input_dfs
            else:
                print(f"Node {self.name} has no results and no input dataframes")
        
        except Exception as e:
            print(f"Error executing {self.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Mark execution as finished
            self.mark_execution_finished()


class ParallelExecutionDataflow(Dataflow):
    """
    A dataflow implementation that executes nodes in parallel as soon as
    their dependencies are satisfied.
    """
    def __init__(self, NodeClass) -> None:
        super().__init__(NodeClass)
        self._final_df_name = "results"
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._execution_lock = threading.Lock()
        self._execution_graph = None
        self._futures = {}
        self._execution_status = {
            'running': False,
            'completed': False,
            'error': None
        }
        self._results = {}
    
    def node(self, name: str, classType=None, *args, **kwargs) -> ParallelExecutionNode:
        """
        Override the node method to use ParallelExecutionNode.
        """
        if name in self.nodes:
            return self.nodes[name]
        else:
            if classType is None or classType == DataflowNode:
                # Use ParallelExecutionNode as the default node type
                self.nodes[name] = ParallelExecutionNode(name, *args, **kwargs)
            elif classType == ParallelProcessingNode:
                # For ParallelProcessingNode, create a ParallelExecutionNode with the same parameters
                is_final = kwargs.get('final', False)
                node = ParallelExecutionNode(name, final=is_final)
                # Copy over the process function if it exists in kwargs
                process_func = kwargs.get('process_func', None)
                if process_func:
                    node.process_func = process_func
                self.nodes[name] = node
            elif issubclass(classType, ParallelExecutionNode):
                # Use the specified class type that extends ParallelExecutionNode
                self.nodes[name] = classType(name, *args, **kwargs)
            else:
                # Use the specified class type
                self.nodes[name] = classType(name, *args, **kwargs)
            
            return self.nodes[name]
    
    def _build_execution_graph(self):
        """
        Build a directed acyclic graph (DAG) of node dependencies for
        determining execution order.
        """
        graph = nx.DiGraph()
        
        # Add all nodes to the graph
        for name, node in self.nodes.items():
            graph.add_node(name, node=node)
        
        # Add dependencies as edges (dependency -> dependent)
        for name, node in self.nodes.items():
            for dep in node._dependencies:
                # Find the name of this dependency node
                dep_name = None
                for n, nd in self.nodes.items():
                    if nd == dep:
                        dep_name = n
                        break
                
                if dep_name:
                    graph.add_edge(dep_name, name)
        
        # Verify this is a DAG (no cycles)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Dataflow contains cyclic dependencies, which is not supported")
        
        return graph
    
    def _get_ready_nodes(self, executed_nodes: Set[str]) -> List[str]:
        """
        Get nodes that are ready to execute because all their dependencies
        have completed.
        """
        ready_nodes = []
        
        for node_name, node_data in self._execution_graph.nodes(data=True):
            # Skip nodes that have already been executed
            if node_name in executed_nodes:
                continue
            
            # Get predecessors (dependencies) of this node
            deps = list(self._execution_graph.predecessors(node_name))
            
            # If all dependencies are executed, this node is ready
            if all(dep in executed_nodes for dep in deps):
                ready_nodes.append(node_name)
        
        return ready_nodes
    
    def execute(self, parameters={}) -> Dict[str, Any]:
        """
        Execute the dataflow in parallel, running nodes as soon as their
        dependencies are satisfied.
        """
        with self._execution_lock:
            # Reset execution status
            self._execution_status = {
                'running': True,
                'completed': False,
                'error': None
            }
            self._futures = {}
            self._results = {}
            
            # Build the execution graph if it doesn't exist
            if self._execution_graph is None:
                self._execution_graph = self._build_execution_graph()
            
            # Initialize set of executed nodes
            executed_nodes = set()
            # Initialize futures dict to track running tasks
            futures = {}
            
            try:
                # Start with nodes that have no dependencies
                ready_nodes = self._get_ready_nodes(executed_nodes)
                
                # Continue until all nodes are executed
                while ready_nodes or futures:
                    # Submit ready nodes for execution
                    for node_name in ready_nodes:
                        node = self.nodes[node_name]
                        future = self._executor.submit(node.parallel_run, parameters)
                        futures[node_name] = future
                    
                    # Clear ready nodes list
                    ready_nodes = []
                    
                    # Wait for at least one task to complete
                    if futures:
                        done, _ = concurrent.futures.wait(
                            list(futures.values()),
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Find completed nodes
                        completed_nodes = []
                        for node_name, future in futures.items():
                            if future.done():
                                # Get result to propagate any exceptions
                                try:
                                    future.result()
                                    print(f"Node {node_name} completed execution successfully")
                                except Exception as e:
                                    self._execution_status['error'] = e
                                    print(f"ERROR in node {node_name}: {e}")
                                    print(f"Cancelling remaining tasks due to error in {node_name}")
                                    raise
                                
                                completed_nodes.append(node_name)
                                executed_nodes.add(node_name)
                        
                        # Remove completed futures
                        for node_name in completed_nodes:
                            del futures[node_name]
                        
                        # Find new ready nodes
                        ready_nodes = self._get_ready_nodes(executed_nodes)
                
                # Collect results from final nodes
                for node_name, node in self.nodes.items():
                    if node._final:
                        results = node.get_results()
                        if results and self._final_df_name in results:
                            data = results[self._final_df_name]
                            if data is not None and hasattr(data, 'values'):
                                self._results[node_name] = data.values.flatten()
            
            except Exception as e:
                self._execution_status['error'] = e
                # Clean up any running futures
                for future in futures.values():
                    future.cancel()
                raise
            
            finally:
                # Update execution status
                self._execution_status['running'] = False
                self._execution_status['completed'] = True
            
            return self._results
    
    def is_running(self) -> bool:
        """Check if the dataflow is currently running."""
        with self._execution_lock:
            return self._execution_status['running']
    
    def is_completed(self) -> bool:
        """Check if the dataflow execution has completed."""
        with self._execution_lock:
            return self._execution_status['completed']
    
    def has_error(self) -> bool:
        """Check if the dataflow execution encountered an error."""
        with self._execution_lock:
            return self._execution_status['error'] is not None
    
    def get_error(self) -> Optional[Exception]:
        """Get the error that occurred during execution, if any."""
        with self._execution_lock:
            return self._execution_status['error']
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of the dataflow execution."""
        with self._execution_lock:
            return self._results
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


class ParallelExecutionDataFlowManager:
    """
    A manager for ParallelExecutionDataflow instances.
    """
    __instance = None
    
    @staticmethod
    def getInstance() -> ParallelExecutionDataFlowManager:
        if ParallelExecutionDataFlowManager.__instance is None:
            ParallelExecutionDataFlowManager()
        return ParallelExecutionDataFlowManager.__instance

    def __init__(self) -> None:
        if ParallelExecutionDataFlowManager.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ParallelExecutionDataFlowManager.__instance = self
            self._dataflows = {}
            self._execution_lock = threading.Lock()
    
    def newDataFlow(self, NodeClass) -> ParallelExecutionDataflow:
        """
        Create or retrieve a dataflow for the given NodeClass.
        """
        # Check if NodeClass is Node class or its subclass
        if not issubclass(NodeClass, Node):
            raise Exception("NodeClass should be a subclass of Node")
        
        # Check if NodeClass is already in the dataFlows
        with self._execution_lock:
            if NodeClass in self._dataflows:
                return self._dataflows[NodeClass]
            
            dataflow = ParallelExecutionDataflow(NodeClass)
            self._dataflows[NodeClass] = dataflow
            return dataflow
    
    def executeDataFlow(self, nodeClass, parameters={}) -> Dict[str, Any]:
        """
        Execute a dataflow synchronously and return the results.
        """
        # Check if nodeClass is subclass of Node
        if not issubclass(nodeClass, Node):
            raise Exception(f"{nodeClass}: NodeClass should be a subclass of Node")
        
        # Check if nodeClass is in the dataFlows
        with self._execution_lock:
            if nodeClass not in self._dataflows:
                raise Exception("NodeClass is not in dataFlows")
            
            # Get the dataflow instance
            dataflow = self._dataflows[nodeClass]
            
            # Execute the dataflow and return the results
            return dataflow.execute(parameters)
    
    def shutdown(self):
        """
        Shutdown all dataflow executors.
        """
        with self._execution_lock:
            for dataflow in self._dataflows.values():
                dataflow.shutdown()
    
    # overload [] operator
    def __getitem__(self, key):
        return self.newDataFlow(key)
