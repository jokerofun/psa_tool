from collections.abc import Callable
from typing import Dict
import pandas as pd
import uuid
import requests
import numpy as np

db_connection = None  # Placeholder for the database connection

class DataflowTask:
    def __init__(self, name: str, final = False) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self._dependencies = []
        self._results = {}
        self._final = final

    def add_dependency(self, node):
        if not isinstance(node, DataflowTask):
            raise TypeError("Dependency must be an instance of DataflowTask class")
        
        self._dependencies.append(node)

    def add_dependencies(self, nodes):
        for node in nodes:
            if not isinstance(node, DataflowTask):
                raise TypeError("Dependency must be an instance of DataflowTask class")
            
            self.add_dependency(node)

    # Overload >> operator dependency chaining
    def __rshift__(self, node):
        node.add_dependency(self)
        return node
    
    def run(self):
        input_dfs = {}
        for node in self._dependencies:
            node.run()
            input_dfs.update(node.get_results())

        print(f"{self.name} task is running...")
        self.process(input_dfs)
        self._results = input_dfs

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        return self._results

class DataFetchingTask(DataflowTask):
    def __init__(self, name: str, source: str = "database.db") -> None:
        super().__init__(name)
        self.source = source

    def fetch_data(self) -> pd.DataFrame:
        """Subclasses must implement this method to fetch data"""
        raise NotImplementedError

    def process(self, dfs: Dict[str, pd.DataFrame]) -> None:
        print(f"Fetching data for {self.name} from {self.source}")
        dfs[self.name] = self.fetch_data()

class DataFetchingFromFileTask(DataFetchingTask):
    def fetch_data(self) -> pd.DataFrame:
        return pd.read_csv(self.source)

class DataFetchingFromDBTask(DataFetchingTask):
    def fetch_data(self) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {self.source}", db_connection)

class DataFetchingFromAPITask(DataFetchingTask):
    def fetch_data(self) -> pd.DataFrame:
        response = requests.get(self.source)
        return pd.DataFrame(response.json())
    
class DataProcessingTask(DataflowTask):
    def __init__(self, name: str, process_func: Callable[[Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]] = None, final = False):
        super().__init__(name, final)
        self.process_func = process_func
    
    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self.process_func(dfs)
    
class MLTask(DataflowTask):
    def __init__(self, name: str, model_func: Callable[[Dict[str, pd.DataFrame]], None], final = True):
        super().__init__(name, final)
        self.model_func = model_func

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.model_func(dfs)

# Example usage
if __name__ == "__main__":
    pricesEUR = DataFetchingFromFileTask(name="pricesEUR", source="data/test_data/pricesEUR.csv")
    pricesDKK = DataFetchingFromFileTask(name="pricesDKK", source="data/test_data/pricesDKK.csv")

    def merge_dataframes(dfs):
        dfs["merged_prices"] = pd.concat([dfs["pricesEUR"], dfs["pricesDKK"]])
        return dfs
    
    def train_model(dfs):
        # Implement model training logic here
        print("Training model")
        return dfs
    
    # Process function can be passed as an argument
    merged = DataProcessingTask(name="merged_prices", process_func=merge_dataframes)
    model = DataProcessingTask(name="model", process_func=None)

    # Or set it later
    model.process_func = train_model
    
    pricesEUR >> merged
    pricesDKK >> merged
    merged >> model
    
    model.run()
    
    print(model.get_results())
