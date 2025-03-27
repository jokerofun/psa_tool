from collections.abc import Callable
import os
from typing import Dict
import pandas as pd
import uuid
import requests
# import dataflow_manager.dataflow_manager as manager

db_connection = None  # Placeholder for the database connection

class DataflowNode:
    def __init__(self, name: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self._dependencies = []
        self._is_executed = False
        self._results = {}

    def add_dependency(self, node):
        self._dependencies.append(node)

    # Overload >> operator
    def __rshift__(self, node):
        node.add_dependency(self)
        return node
    
    def getResults(self) -> Dict[str, pd.DataFrame]:
        return self._results
    
    def run(self):
        if not self._is_executed:
            for node in self._dependencies:
                node.run()
            input_dfs = {}            
            for node in self._dependencies:
                ## append node name to the keys added
                for key, value in node.getResults().items():
                    input_dfs[key] = value
            print(self.name + " is running")
            # print(input_dfs)
            self._results = self.process(input_dfs)
            self._is_executed = True

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses should implement this method")


class DataProcessingNode(DataflowNode):
    def __init__(self, name: str, process_func: Callable[[Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]]) -> None:
        super().__init__(name)
        self.process_func = process_func
    
    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        dfs = self.process_func(dfs)
        return dfs


class DataFetchingFromFileNode(DataflowNode):
    def __init__(self, name: str, file_path: str | os.PathLike) -> None:
        super().__init__(name)
        self.file_path = file_path

    def process(self, dfs):
        # Implement data fetching logic here
        print(f"Fetching data from file: {self.file_path}")
        # Example fetching: read data from file
        dfs[self.name] = pd.read_csv(self.file_path)
        return dfs


class DataFetchingFromDBNode(DataflowNode):
    def __init__(self, name: str, table_name: str) -> None:
        super().__init__(name)
        self.table_name = table_name

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Implement data fetching logic here
        print(f"Fetching data from DB table: {self.table_name}")
        # Example fetching: query data from a database
        dfs[self.name] = pd.read_sql(
            f"SELECT * FROM {self.table_name}", db_connection)
        return dfs


class DataFetchingFromAPINode(DataflowNode):
    def __init__(self, name: str, api_url: str) -> None:
        super().__init__(name)
        self.api_url = api_url

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Implement data fetching logic here
        print(f"Fetching data from API: {self.api_url}")
        # Example fetching: call an API to get data
        response = requests.get(self.api_url)
        dfs[self.name] = pd.DataFrame(response.json())
        return dfs     
    
    
# Example usage
if __name__ == "__main__":
    pricesEUR = DataFetchingFromFileNode(name="pricesEUR", file_path="dataflow_manager/test_data/pricesEUR.csv")
    pricesDKK = DataFetchingFromFileNode(name="pricesDKK", file_path="dataflow_manager/test_data/pricesDKK.csv")

    def merge_dataframes(dfs):
        dfs["merged_prices"] = pd.concat([dfs["pricesEUR"], dfs["pricesDKK"]])
        return dfs
    
    def train_model(dfs):
        # Implement model training logic here
        print("Training model")
        return dfs
    
    merged = DataProcessingNode(name="merged_prices", process_func=merge_dataframes)
    model = DataProcessingNode(name="model", process_func=train_model)
    
    pricesEUR >> merged
    pricesDKK >> merged
    merged >> model
    
    model.run()
    
    print(model.getResults())
    
    # data_node = DataProcessingNode("data_node", "sample_data")
    # result_dfs = data_node.execute(dfs)
    # print(result_dfs)
