from .dataflow_classes import DataflowNode, DataProcessingNode
import pandas as pd
from typing import Dict, Any, Callable

class ParallelDataflowNode(DataflowNode):
    """
    Enhanced version of DataflowNode that correctly stores results from processing.
    This fixes the issue in the original DataflowNode.run() method.
    """
    def __init__(self, name: str, final=False) -> None:
        super().__init__(name, final)

    def run(self, parameters={}):
        """
        Override the run method to properly store the results from processing.
        """
        # Collect results from dependencies
        input_dfs = {}
        for node in self._dependencies:
            node.run(parameters)
            # Get results from dependencies
            input_dfs.update(node.get_results())

        print(f"{self.name} is running")
        # Process the inputs and STORE the results (this is the key fix)
        result_dfs = self.process(input_dfs, parameters)
        # Only update _results if process returned something
        if result_dfs is not None:
            self._results = result_dfs
        # If process returned None but input_dfs has content, use that
        elif not self._results and input_dfs:
            self._results = input_dfs


class ParallelProcessingNode(ParallelDataflowNode):
    """
    Enhanced version of DataProcessingNode with proper results handling.
    """
    def __init__(self, name: str, process_func: Callable[[Dict[str, pd.DataFrame], Dict], Dict[str, pd.DataFrame]] = None, final=False):
        super().__init__(name, final)
        self.process_func = process_func

    def process(self, dfs: Dict[str, pd.DataFrame], parameters={}) -> Dict[str, pd.DataFrame]:
        """
        Process the input DataFrames using the provided process_func.
        """
        if self.process_func:
            return self.process_func(dfs, parameters)
        return dfs
