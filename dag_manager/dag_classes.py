# import pandas as pd
# import uuid
# import requests

# db_connection = None  # Placeholder for the database connection

# class DAGNode:
#     def __init__(self, name):
#         self.id = uuid.uuid4()
#         self.name = name

#     def execute(self, df):
#         raise NotImplementedError("Subclasses should implement this method")

# class DataProcessingNode(DAGNode):
#     def __init__(self, name, data):
#         super().__init__(name)
#         self.data = data

#     def execute(self, df):
#         # Implement data processing logic here
#         print(f"Processing data: {self.data}")
#         # Example processing: return the dataframe as is
#         return df

# class DataFetchingFromFileNode(DAGNode):
#     def __init__(self, name, file_path):
#         super().__init__(name)
#         self.file_path = file_path

#     def execute(self, df):
#         # Implement data fetching logic here
#         print(f"Fetching data from file: {self.file_path}")
#         # Example fetching: read data from file
#         return pd.read_csv(self.file_path)

# class DataFetchingFromDBNode(DAGNode):
#     def __init__(self, name, table_name):
#         super().__init__(name)
#         self.table_name = table_name

#     def execute(self, df):
#         # Implement data fetching logic here
#         print(f"Fetching data from DB table: {self.table_name}")
#         # Example fetching: query data from a database
#         return pd.read_sql(f"SELECT * FROM {self.table_name}", db_connection)

# class DataFetchingFromAPINode(DAGNode):
#     def __init__(self, name, api_url):
#         super().__init__(name)
#         self.api_url = api_url

#     def execute(self, df):
#         # Implement data fetching logic here
#         print(f"Fetching data from API: {self.api_url}")
#         # Example fetching: call an API to get data
#         response = requests.get(self.api_url)
#         return pd.DataFrame(response.json())

# # Example usage
# if __name__ == "__main__":
#     df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
#     data_node = DataProcessingNode("data_node", "sample_data")
#     result_df = data_node.execute(df)
#     print(result_df)


import pandas as pd
import uuid
import requests

db_connection = None  # Placeholder for the database connection

class DAGNode:
    def __init__(self, name):
        self.id = uuid.uuid4()
        self.name = name

    def execute(self, dfs):
        raise NotImplementedError("Subclasses should implement this method")

class DataProcessingNode(DAGNode):
    def __init__(self, name, data):
        super().__init__(name)
        self.data = data

    def execute(self, dfs):
        # Implement data processing logic here
        print(f"Processing data: {self.data}")
        # Example processing: return the first dataframe as is
        return dfs

class DataFetchingFromFileNode(DAGNode):
    def __init__(self, name, file_path):
        super().__init__(name)
        self.file_path = file_path

    def execute(self, dfs):
        # Implement data fetching logic here
        print(f"Fetching data from file: {self.file_path}")
        # Example fetching: read data from file
        dfs[self.name] = pd.read_csv(self.file_path)
        return dfs

class DataFetchingFromDBNode(DAGNode):
    def __init__(self, name, table_name):
        super().__init__(name)
        self.table_name = table_name

    def execute(self, dfs):
        # Implement data fetching logic here
        print(f"Fetching data from DB table: {self.table_name}")
        # Example fetching: query data from a database
        dfs[self.name] = pd.read_sql(f"SELECT * FROM {self.table_name}", db_connection)
        return dfs

class DataFetchingFromAPINode(DAGNode):
    def __init__(self, name, api_url):
        super().__init__(name)
        self.api_url = api_url

    def execute(self, dfs):
        # Implement data fetching logic here
        print(f"Fetching data from API: {self.api_url}")
        # Example fetching: call an API to get data
        response = requests.get(self.api_url)
        dfs[self.name] = pd.DataFrame(response.json())
        return dfs

# Example usage
if __name__ == "__main__":
    dfs = {"initial_df": pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})}
    data_node = DataProcessingNode("data_node", "sample_data")
    result_dfs = data_node.execute(dfs)
    print(result_dfs)