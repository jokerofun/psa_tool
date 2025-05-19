from collections.abc import Callable
from typing import Dict
import pandas as pd
import uuid
import requests

db_connection = None  # Placeholder for the database connection

class DataflowNode:
    def __init__(self, name: str, final = False) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self._dependencies = []
        self._results = {}
        self._final = final

    def add_dependency(self, node):
        self._dependencies.append(node)

    # Overload >> operator dependency chaining
    def __rshift__(self, node):
        node.add_dependency(self)
        return node
    
    def run(self):
        input_dfs = {}
        for node in self._dependencies:
            node.run()
            input_dfs.update(node.get_results())

        print(f"{self.name} is running")
        self.process(input_dfs)
        self._results = input_dfs

    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        return self._results


class DataProcessingNode(DataflowNode):
    def __init__(self, name: str, process_func: Callable[[Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]] = None, final = False):
        super().__init__(name, final)
        self.process_func = process_func
    
    def process(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self.process_func(dfs)
    
class DataFetchingNode(DataflowNode):
    def __init__(self, name: str, source: str) -> None:
        super().__init__(name)
        self.source = source

    def fetch_data(self) -> pd.DataFrame:
        """Subclasses must implement this method to fetch data"""
        raise NotImplementedError

    def process(self, dfs: Dict[str, pd.DataFrame]) -> None:
        print(f"Fetching data for {self.name} from {self.source}")
        dfs[self.name] = self.fetch_data()


class DataFetchingFromFileNode(DataFetchingNode):
    def fetch_data(self) -> pd.DataFrame:
        return pd.read_csv(self.source)


class DataFetchingFromDBNode(DataFetchingNode):
    def fetch_data(self) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {self.source}", db_connection)


class DataFetchingFromAPINode(DataFetchingNode):
    def fetch_data(self) -> pd.DataFrame:
        response = requests.get(self.source)
        return pd.DataFrame(response.json())
    
    
# Example usage
if __name__ == "__main__":
    pricesEUR = DataFetchingFromFileNode(name="pricesEUR", source="dataflow_manager/test_data/pricesEUR.csv")
    pricesDKK = DataFetchingFromFileNode(name="pricesDKK", source="dataflow_manager/test_data/pricesDKK.csv")

    def merge_dataframes(dfs):
        dfs["merged_prices"] = pd.concat([dfs["pricesEUR"], dfs["pricesDKK"]])
        return dfs
    
    def train_model(dfs):
        # Implement model training logic here
        print("Training model")
        return dfs
    
    # Process function can be passed as an argument
    merged = DataProcessingNode(name="merged_prices", process_func=merge_dataframes)
    model = DataProcessingNode(name="model", process_func=None)

    # Or set it later
    model.process_func = train_model
    
    pricesEUR >> merged
    pricesDKK >> merged
    merged >> model
    
    model.run()
    
    print(model.get_results())
