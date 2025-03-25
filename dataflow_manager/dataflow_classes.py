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
        # manager.addNode(self)

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

class DependencyNode():
    def __init__(self, name: str):
        self._dependencies = []
        self._is_executed = False
        self._results = {}
    
        self._name = name

    def add_dependency(self, node):
        self._dependencies.append(node)
    
    # overlode >> operator
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
                    input_dfs[node._name + "_" + key] = value
            print(self._name + " is running")
            print(input_dfs)
            self._results = self.process(input_dfs)
            self._is_executed = True
    
    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return dfs      
    
                
    
    
# Example usage
if __name__ == "__main__":
    dfs = {"initial_df": pd.DataFrame(
        {"column1": [1, 2, 3], "column2": [4, 5, 6]})}
    dfs2 = {"initial_df2": pd.DataFrame(
        {"column1": [10, 20, 30], "column2": [40, 50, 60]})}
    dep1 = DependencyNode("csv1")
    dep2 = DependencyNode("csv2")
    dep3 = DependencyNode("preproc")
    dep4 = DependencyNode("model")
    
    dep1 >> dep3
    dep2 >> dep3
    dep3 >> dep4
    
    dep1.process = lambda dfs: {"csv1": pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})}
    dep2.process = lambda dfs: {"csv2": pd.DataFrame({"column1": [10, 20, 30], "column2": [40, 50, 60]})}
    # merge incoming dataframes
    dep3.process = lambda dfs: {"preproc": pd.concat([dfs["csv1_csv1"], dfs["csv2_csv2"]])}
    # double
    dep4.process = lambda dfs: {"model": dfs["preproc_preproc"]}
    
    dep4.run()
    
    print(dep4.getResults())
    
    # data_node = DataProcessingNode("data_node", "sample_data")
    # result_dfs = data_node.execute(dfs)
    # print(result_dfs)
