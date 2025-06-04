# filepath: /home/sobibence/AAU/3_semester/project/psa_tool/src/dataflow/dataflow_manager_paralell.py
"""
Parallel implementation of the DataflowManager that executes dataflows concurrently.
"""
from __future__ import annotations
import concurrent.futures
import multiprocessing

from .dataflow_v2 import Dataflow
from src.optimization.base_domain import Node


# Simplified implementation of ParallelDataFlowManager
class ParallelDataFlowManager:
    """
    A parallel implementation of the DataflowManager that executes dataflows concurrently
    using thread pools. This provides better performance for IO-bound workloads.
    """

    def __init__(self):
        self._dataflows = {}
        # Use integer division to ensure max_workers is an integer
        max_workers = max(1, multiprocessing.cpu_count())
        # max_workers = 8
        # print(f"Creating ThreadPoolExecutor with {max_workers} workers")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )

    def new_dataflow(self, object, dataflow=None) -> Dataflow:
        """
        Create a new dataflow for the given object or return an existing one.
        This method provides compatibility with DataflowManager_v2 interface.
        """
        if object is None:
            raise Exception("Object cannot be None")
        if not isinstance(object, Node):
            raise Exception("Object should be a subclass of Node")
        if dataflow is not None and not isinstance(dataflow, Dataflow):
            raise Exception("Dataflow should be an instance of Dataflow class")
            
        # Use object.name as key - matching the DataflowManager_v2 interface
        if object.name in self._dataflows:
            return self._dataflows[object.name]
            
        if dataflow is None:
            dataflow = Dataflow(object.name, object)
            
        self._dataflows[object.name] = dataflow
        return dataflow

    def execute(self) -> None:
        """
        Execute all dataflows in parallel using ThreadPoolExecutor.
        This method provides compatibility with DataflowManager_v2 interface.
        """
        futures = []
        for dataflow in self._dataflows.values():
            # Submit each dataflow's execution to the thread pool
            # check if executor is shutdow
            futures.append(self._executor.submit(dataflow.execute))
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    def get_data(self, object, task_name):
        """
        Get data from a specific task in an object's dataflow.
        This method provides compatibility with DataflowManager_v2 interface.
        """
        if not isinstance(object, Node):
            raise Exception("Object should be a subclass of Node")
            
        if object.name not in self._dataflows:
            raise Exception(f"Object {object.name} doesn't have any dataflow")
            
        dataflow = self._dataflows[object.name]
        return dataflow.get_data(task_name)
    
    # Overload [] operator for compatibility
    def __getitem__(self, key):
        """Get a dataflow by key (object)"""
        return self.new_dataflow(key)
    
    def reset(self) -> None:
        """
        Reset the dataflow manager, clearing all dataflows.
        This method provides compatibility with DataflowManager_v2 interface.
        """
        self._dataflows.clear()
        executor = self._executor
        executor.shutdown(wait=True)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers= max(1, multiprocessing.cpu_count())
        )
        # self.__instance = None
        # print("ParallelDataFlowManager has been reset.")
