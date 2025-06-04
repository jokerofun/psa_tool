from __future__ import annotations
from enum import Enum
from typing import Optional, Union
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Import all dataflow manager implementations
from src.dataflow.dataflow_manager_v2 import DataflowManager
from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
# We'll need to implement a distributed dataflow manager


# Global setting that can be changed by the application
GLOBAL_EXECUTION_MODE = "serial"



class DataflowFactory:
    """
    Factory class responsible for providing the appropriate DataflowManager implementation
    based on the specified execution mode.
    """
    _instance: Optional[Union[DataflowManager, ParallelDataFlowManager]] = None
    _current_mode = "serial"
    
    @classmethod
    def get_dataflow_manager(cls) -> Union[DataflowManager, ParallelDataFlowManager]:
        """
        Get the appropriate DataflowManager instance based on the current execution mode.
        """
        # If we don't have an instance or the execution mode has changed
        if cls._instance is None or cls._current_mode != GLOBAL_EXECUTION_MODE:
            # Reset any existing singleton instances
            DataflowManager._instance = None
            ParallelDataFlowManager._instance = None
            
            # Create the appropriate manager based on the mode
            if GLOBAL_EXECUTION_MODE == "serial":
                cls._instance = DataflowManager()
                # print("Factory providing SERIAL DataflowManager")
            elif GLOBAL_EXECUTION_MODE == "parallel":
                cls._instance = ParallelDataFlowManager()
                # print("Factory providing PARALLEL DataflowManager")
            else:
                raise ValueError(f"Unknown execution mode: {GLOBAL_EXECUTION_MODE}")
            cls._current_mode = GLOBAL_EXECUTION_MODE
            
        return cls._instance
    
    @classmethod
    def set_execution_mode(cls, mode) -> None:
        """
        Set the global execution mode.
        Next call to get_instance will return the appropriate manager type.
        """
        global GLOBAL_EXECUTION_MODE
        GLOBAL_EXECUTION_MODE = mode
        
        # Reset instance so it will be recreated with the new mode
        cls._instance = None
        
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the current DataflowManager instance.
        This is useful for testing or when the execution mode changes.
        """
        if cls._instance is not None:
            cls._instance.reset()
            
        cls._instance = None
        cls._current_mode = "serial"
        # also reset the singleton instances of the managers
        
        
    


# Convenience function to get a dataflow manager with the current execution mode
def get_dataflow_manager() -> Union[DataflowManager, ParallelDataFlowManager]:
    """
    Get a DataflowManager instance configured according to the global execution mode.
    Returns different manager types based on the current execution mode.
    This is the main function to be used by clients.
    """
    return DataflowFactory.get_dataflow_manager()


def get_dataflow_constructor(name, object_ref):
    return DataflowFactory.get_dataflow_manager().new_dataflow(name, object_ref)