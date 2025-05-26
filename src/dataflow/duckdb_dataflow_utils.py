import duckdb
import pandas as pd
import os
import uuid
from typing import Dict, Optional, List
import threading

class DuckDBDataflowManager:
    """
    A utility class for managing dataflow data persistence with DuckDB.
    This class handles storing and retrieving dataframes used by dataflow nodes.
    """
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance(db_path: str = None) -> 'DuckDBDataflowManager':
        """
        Get the singleton instance of the DuckDBDataflowManager.
        
        Args:
            db_path: Path to the DuckDB database file. If None, uses the default path.
        
        Returns:
            The singleton instance of DuckDBDataflowManager
        """
        with DuckDBDataflowManager._lock:
            if DuckDBDataflowManager._instance is None:
                DuckDBDataflowManager._instance = DuckDBDataflowManager(db_path)
            return DuckDBDataflowManager._instance
    
    def __init__(self, db_path: str = None):
        """
        Initialize the DuckDBDataflowManager.
        
        Args:
            db_path: Path to the DuckDB database file. If None, uses the default path.
        """
        # Default to a 'dataflow_data' directory in the project data folder
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'duckdb')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'dataflow.duckdb')
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._execution_ids = {}
        self._conn_lock = threading.Lock()
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize the database schema for dataflow operations."""
        with self._conn_lock:
            # Create a table to track dataflow executions
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dataflow_executions (
                    execution_id VARCHAR PRIMARY KEY,
                    dataflow_name VARCHAR NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR DEFAULT 'running',  -- 'running', 'completed', 'error'
                    parameters VARCHAR   -- JSON serialized parameters
                )
            """)
            
            # Create a table to store node results
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS node_results (
                    result_id VARCHAR PRIMARY KEY,
                    execution_id VARCHAR NOT NULL,
                    node_name VARCHAR NOT NULL,
                    table_name VARCHAR NOT NULL,  -- Name of the table where the actual data is stored
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES dataflow_executions(execution_id)
                )
            """)
            
            self.conn.commit()
    
    def start_execution(self, dataflow_name: str, parameters: Dict = None) -> str:
        """
        Start a new dataflow execution and get an execution ID.
        
        Args:
            dataflow_name: Name of the dataflow
            parameters: Optional parameters for the dataflow execution
        
        Returns:
            str: The execution ID
        """
        execution_id = str(uuid.uuid4())
        
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO dataflow_executions (execution_id, dataflow_name, parameters) VALUES (?, ?, ?)",
                (execution_id, dataflow_name, str(parameters) if parameters else None)
            )
            self.conn.commit()
        
        return execution_id
    
    def complete_execution(self, execution_id: str, status: str = 'completed'):
        """
        Mark a dataflow execution as complete.
        
        Args:
            execution_id: The execution ID to complete
            status: The final status ('completed' or 'error')
        """
        with self._conn_lock:
            self.conn.execute(
                "UPDATE dataflow_executions SET end_time = CURRENT_TIMESTAMP, status = ? WHERE execution_id = ?",
                (status, execution_id)
            )
            self.conn.commit()
    
    def store_dataframe(self, execution_id: str, node_name: str, df_name: str, df: pd.DataFrame) -> str:
        """
        Store a dataframe result from a node execution.
        
        Args:
            execution_id: The execution ID this result belongs to
            node_name: The name of the node that produced this result
            df_name: The name given to this dataframe by the node
            df: The pandas DataFrame to store
        
        Returns:
            str: The result ID
        """
        result_id = str(uuid.uuid4())
        # Make table name safe for SQL
        safe_name = ''.join(char for char in df_name if char.isalnum())
        table_name = f"df_{execution_id.replace('-', '')}_{node_name}_{safe_name}"
        
        # Check if DataFrame is empty
        if df.empty:
            df = pd.DataFrame({'dummy': [0]})  # Add a dummy row to avoid empty table errors
            
        with self._conn_lock:
            try:
                # Create a table for this specific dataframe
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Register the dataframe with DuckDB
                self.conn.register("temp_df", df)
                
                # Create the table from the registered dataframe
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
                
                # Record the result metadata
                self.conn.execute(
                    "INSERT INTO node_results (result_id, execution_id, node_name, table_name) VALUES (?, ?, ?, ?)",
                    (result_id, execution_id, node_name, table_name)
                )
                self.conn.commit()
                print(f"Successfully stored dataframe '{df_name}' with {len(df)} rows")
            except Exception as e:
                print(f"Error storing dataframe: {e}")
                # Create a simpler table if the first attempt failed
                try:
                    self.conn.execute(f"CREATE TABLE {table_name} (id INTEGER)")
                    self.conn.execute(f"INSERT INTO {table_name} VALUES (1)")
                    
                    self.conn.execute(
                        "INSERT INTO node_results (result_id, execution_id, node_name, table_name) VALUES (?, ?, ?, ?)",
                        (result_id, execution_id, node_name, table_name)
                    )
                    self.conn.commit()
                    print(f"Created backup table for '{df_name}'")
                except Exception as e2:
                    print(f"Error creating backup table: {e2}")
        
        return result_id
    
    def store_node_results(self, execution_id: str, node_name: str, 
                           results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Store multiple dataframe results from a node execution.
        
        Args:
            execution_id: The execution ID these results belong to
            node_name: The name of the node that produced these results
            results: Dictionary of dataframes to store (name -> DataFrame)
        
        Returns:
            Dict[str, str]: Dictionary mapping dataframe names to result IDs
        """
        result_ids = {}
        for df_name, df in results.items():
            result_id = self.store_dataframe(execution_id, node_name, df_name, df)
            result_ids[df_name] = result_id
        
        return result_ids
    
    def load_node_results(self, execution_id: str, node_name: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load all result dataframes for a node or all nodes in an execution.
        
        Args:
            execution_id: The execution ID to load results for
            node_name: Optional node name to filter results
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of all dataframes
        """
        with self._conn_lock:
            if node_name:
                # Get results for a specific node
                result = self.conn.execute(
                    "SELECT table_name FROM node_results WHERE execution_id = ? AND node_name = ?",
                    (execution_id, node_name)
                ).fetchall()
            else:
                # Get results for all nodes
                result = self.conn.execute(
                    "SELECT table_name FROM node_results WHERE execution_id = ?",
                    (execution_id,)
                ).fetchall()
                
            dfs = {}
            for row in result:
                try:
                    table_name = row[0]
                    # Extract the dataframe name from the table name
                    df_name = table_name.split('_')[-1]
                    
                    # Check if table exists
                    table_exists = self.conn.execute(
                        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                    ).fetchone()[0]
                    
                    if table_exists:
                        df = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                        # Remove dummy column if it exists
                        if 'dummy' in df.columns and len(df.columns) == 1:
                            df = pd.DataFrame()
                        dfs[df_name] = df
                except Exception as e:
                    print(f"Error loading table {row[0] if row else 'unknown'}: {e}")
            
            return dfs
    
    def get_dependency_results(self, execution_id: str, dependency_nodes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get all result dataframes from dependency nodes.
        
        Args:
            execution_id: The execution ID to load results from
            dependency_nodes: List of node names that are dependencies
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of all dependency dataframes
        """
        all_dfs = {}
        for node_name in dependency_nodes:
            node_dfs = self.load_node_results(execution_id, node_name)
            all_dfs.update(node_dfs)
        
        return all_dfs

    def clear_execution_data(self, execution_id: str):
        """
        Remove all data associated with an execution.
        
        Args:
            execution_id: The execution ID to clear
        """
        with self._conn_lock:
            # Find all tables for this execution
            result = self.conn.execute(
                "SELECT table_name FROM node_results WHERE execution_id = ?",
                (execution_id,)
            ).fetchall()
            
            # Drop each table
            for row in result:
                table_name = row[0]
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Delete metadata records
            self.conn.execute("DELETE FROM node_results WHERE execution_id = ?", (execution_id,))
            self.conn.execute("DELETE FROM dataflow_executions WHERE execution_id = ?", (execution_id,))
            self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
