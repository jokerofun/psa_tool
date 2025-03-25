import os
from typing import Dict
import pandas as pd
import uuid
import requests
import dataflow_manager.dataflow_manager as manager

db_connection = None  # Placeholder for the database connection

class DataflowNode:
    def __init__(self, name: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        manager.addNode(self)

    def execute(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses should implement this method")


class DataProcessingNode(DataflowNode):
    def __init__(self, name: str, data: pd.DataFrame, data_name: str) -> None:
        super().__init__(name)
        self.data = data
        self.data_name = data_name

    def execute(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.data is not None:
            dfs[self.name] = self.process(self.data)
        else:
            dfs[self.name] = self.process(dfs[self.data_name])
            
        return dfs
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement data processing logic here
        print(f"Processing data: {data}")
        # Example processing: return the first dataframe as is
        return data


class DataFetchingFromFileNode(DataflowNode):
    def __init__(self, name: str, file_path: str | os.PathLike) -> None:
        super().__init__(name)
        self.file_path = file_path

    def execute(self, dfs):
        # Implement data fetching logic here
        print(f"Fetching data from file: {self.file_path}")
        # Example fetching: read data from file
        dfs[self.name] = pd.read_csv(self.file_path)
        return dfs


class DataFetchingFromDBNode(DataflowNode):
    def __init__(self, name: str, table_name: str) -> None:
        super().__init__(name)
        self.table_name = table_name

    def execute(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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

    def execute(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Implement data fetching logic here
        print(f"Fetching data from API: {self.api_url}")
        # Example fetching: call an API to get data
        response = requests.get(self.api_url)
        dfs[self.name] = pd.DataFrame(response.json())
        return dfs


# Example usage
if __name__ == "__main__":
    dfs = {"initial_df": pd.DataFrame(
        {"column1": [1, 2, 3], "column2": [4, 5, 6]})}
    data_node = DataProcessingNode("data_node", "sample_data")
    result_dfs = data_node.execute(dfs)
    print(result_dfs)
