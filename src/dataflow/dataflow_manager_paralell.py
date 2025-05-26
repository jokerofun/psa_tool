# filepath: /home/sobibence/AAU/3_semester/project/psa_tool/src/dataflow/dataflow_manager_paralell.py
from __future__ import annotations
import concurrent.futures
from typing import Dict, Any, Optional
import threading
import time

from .dataflow import Dataflow
from .parallel_dataflow_classes import ParallelDataflowNode, ParallelProcessingNode
from .dataflow_classes import DataProcessingNode
from src.optimization.solver_classes import Node

class ParallelDataflow(Dataflow):
    """
    Enhanced version of Dataflow that tracks execution status and results
    for parallel processing.
    """
    def __init__(self, NodeClass) -> None:
        super().__init__(NodeClass)
        self._is_running = False
        self._is_completed = False
        self._result = None
        self._execution_lock = threading.Lock()
        self._error = None
        self._last_run_params = {}
    
    def node(self, name: str, classType=None, *args, **kwargs) -> None:
        """
        Override the node method to use our enhanced node classes.
        """
        if name in self.nodes:
            return self.nodes[name]
        else:
            if classType is None:
                # Use ParallelProcessingNode instead of DataProcessingNode
                self.nodes[name] = ParallelProcessingNode(name, *args, **kwargs)
            elif classType == DataProcessingNode:
                # Replace DataProcessingNode with ParallelProcessingNode
                self.nodes[name] = ParallelProcessingNode(name, *args, **kwargs)
            else:
                # Use the specified class type
                self.nodes[name] = classType(name, *args, **kwargs)
            return self.nodes[name]
    
    def run_async(self, parameters: dict) -> None:
        """Start asynchronous execution of the dataflow with given parameters."""
        with self._execution_lock:
            self._is_running = True
            self._is_completed = False
            self._result = None
            self._error = None
            self._last_run_params = parameters.copy() if parameters else {}
        
        try:
            # Find the node marked as final
            final_node = None
            for node in self.nodes.values():
                if node._final:
                    final_node = node
                    break
            
            if final_node:
                # Execute the final node which will trigger the dependency chain
                # This will now use the fixed run implementation from ParallelDataflowNode
                final_node.run(parameters)
                
                # Get all results from the final node
                results = final_node.get_results()
                
                # Check if we have any results
                if results:
                    # Try to get the results dataframe 
                    if self._final_df_name in results:
                        data = results[self._final_df_name]
                        if data is not None and hasattr(data, 'values'):
                            self._result = data.values.flatten()
                    else:
                        # If the named result isn't there, try to use the first result we find
                        for key, data in results.items():
                            if data is not None and hasattr(data, 'values'):
                                self._result = data.values.flatten()
                                print(f"Using result key: {key}")
                                break
                else:
                    print("Warning: No results returned from dataflow processing")
            else:
                self._error = ValueError("No final node found in the dataflow")
                
        except Exception as e:
            self._error = e
            import traceback
            print(f"Error in run_async: {e}")
            print(traceback.format_exc())
        finally:
            with self._execution_lock:
                self._is_running = False
                self._is_completed = True
    
    @property
    def is_running(self) -> bool:
        """Check if the dataflow is currently running."""
        with self._execution_lock:
            return self._is_running
    
    @property
    def is_completed(self) -> bool:
        """Check if the dataflow execution has completed."""
        with self._execution_lock:
            return self._is_completed
    
    @property
    def has_error(self) -> bool:
        """Check if the dataflow execution encountered an error."""
        with self._execution_lock:
            return self._error is not None
    
    def get_error(self) -> Optional[Exception]:
        """Get the error that occurred during execution, if any."""
        with self._execution_lock:
            return self._error
    
    def get_result(self) -> Any:
        """Get the result of the dataflow execution."""
        with self._execution_lock:
            return self._result
    
    def wait_for_completion(self, timeout=None) -> bool:
        """
        Wait for the dataflow execution to complete.
        Returns True if completed, False if timed out.
        """
        start_time = time.time()
        while True:
            with self._execution_lock:
                if self._is_completed:
                    return True
            
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            time.sleep(0.1)


# Singleton class for parallel dataflow management
class ParallelDataFlowManager:
    __instance = None
    
    @staticmethod
    def getInstance() -> ParallelDataFlowManager:
        if ParallelDataFlowManager.__instance is None:
            ParallelDataFlowManager()
        return ParallelDataFlowManager.__instance

    def __init__(self) -> None:
        if ParallelDataFlowManager.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ParallelDataFlowManager.__instance = self
            self._dataflows = {}
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
            self._futures = {}
            self._lock = threading.Lock()

    def newDataFlow(self, NodeClass) -> ParallelDataflow:
        """
        Create or retrieve a dataflow for the given NodeClass.
        """
        # Check if NodeClass is Node class or its subclass
        if not issubclass(NodeClass, Node):
            raise Exception("NodeClass should be a subclass of Node")
        
        # Check if NodeClass is already in the dataFlows
        with self._lock:
            if NodeClass in self._dataflows:
                return self._dataflows[NodeClass]
            
            dataflow = ParallelDataflow(NodeClass)
            self._dataflows[NodeClass] = dataflow
            return dataflow

    def executeDataFlow(self, nodeClass, parameters={}) -> str:
        """
        Execute a dataflow asynchronously and return a task ID for checking status later.
        """
        # Check if nodeClass is subclass of Node
        if not issubclass(nodeClass, Node):
            raise Exception(f"{nodeClass}: NodeClass should be a subclass of Node")
        
        # Check if nodeClass is in the dataFlows
        with self._lock:
            if nodeClass not in self._dataflows:
                raise Exception("NodeClass is not in dataFlows")
            
            # Get the dataflow instance
            dataflow = self._dataflows[nodeClass]
            
            # Create a unique task ID
            task_id = f"{nodeClass.__name__}_{time.time()}"
            
            # Submit the task to the executor
            future = self._executor.submit(dataflow.run_async, parameters)
            self._futures[task_id] = (future, dataflow)
            
            return task_id

    def isCompleted(self, task_id: str) -> bool:
        """
        Check if the task with the given ID has completed.
        """
        with self._lock:
            if task_id not in self._futures:
                raise Exception(f"Task ID {task_id} not found")
            
            future, dataflow = self._futures[task_id]
            # The future is done but we also need to check if the dataflow has completed
            return future.done() and dataflow.is_completed

    def getData(self, task_id: str, wait: bool = False, timeout: float = None):
        """
        Get data from a completed task.
        If wait is True, wait for the task to complete.
        If timeout is specified, wait up to that many seconds.
        """
        with self._lock:
            if task_id not in self._futures:
                raise Exception(f"Task ID {task_id} not found")
            
            future, dataflow = self._futures[task_id]
        
        if wait:
            # Wait for the future to complete
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Task {task_id} did not complete within the specified timeout")
            
            # Wait for the dataflow to be marked as completed
            if not dataflow.wait_for_completion(timeout):
                raise TimeoutError(f"Dataflow for task {task_id} did not complete within the specified timeout")
        
        elif not self.isCompleted(task_id):
            raise Exception(f"Task {task_id} is not yet completed")
        
        # Check if there was an error
        if dataflow.has_error:
            error = dataflow.get_error()
            raise Exception(f"Task {task_id} failed with error: {error}")
        
        # Return the result
        return dataflow.get_result()

    def cancelTask(self, task_id: str) -> bool:
        """
        Cancel a running task.
        Returns True if cancellation was successful, False otherwise.
        """
        with self._lock:
            if task_id not in self._futures:
                raise Exception(f"Task ID {task_id} not found")
            
            future, _ = self._futures[task_id]
            return future.cancel()

    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor.
        If wait is True, wait for all running tasks to complete.
        """
        self._executor.shutdown(wait=wait)
    
    # Convenience method to get task status
    def getTaskStatus(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        Returns a dictionary with the task status.
        """
        with self._lock:
            if task_id not in self._futures:
                raise Exception(f"Task ID {task_id} not found")
            
            future, dataflow = self._futures[task_id]
            
            return {
                "task_id": task_id,
                "is_running": dataflow.is_running,
                "is_completed": dataflow.is_completed,
                "has_error": dataflow.has_error,
                "is_cancelled": future.cancelled(),
            }
    
    # overload [] operator
    def __getitem__(self, key):
        return self.newDataFlow(key)