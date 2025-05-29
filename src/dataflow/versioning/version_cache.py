#!/usr/bin/env python3
"""
Version Cache Manager: Manages caching of dataframe processing results.

This module provides a cache manager that:
1. Stores versioned results in DuckDB
2. Retrieves cached results based on version keys
3. Handles version metadata and cleanup
"""

import duckdb
import pandas as pd
import os
import json
import datetime
import time
from typing import Dict, Any, List, Optional, Callable, Tuple

from .version_utils import generate_version_key, are_dataframes_similar

class DuckDBVersionCache:
    """
    A cache manager for versioned dataframes using DuckDB.
    
    This class provides methods to:
    1. Store dataframes with version information
    2. Retrieve dataframes based on version keys
    3. Manage version metadata
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the DuckDB version cache.
        
        Args:
            db_path: Path to the DuckDB database file. If None, uses a default path.
        """
        # Default to 'versions' directory in the project data folder
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), 'data', 'duckdb')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'version_cache.duckdb')
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize the database schema for version caching."""
        # Try to drop existing tables if they exist (for a fresh start)
        try:
            self.conn.execute("DROP TABLE IF EXISTS version_results")
            self.conn.execute("DROP TABLE IF EXISTS version_metadata")
        except:
            pass
            
        # Create a table to track versions
        self.conn.execute("""
            CREATE TABLE version_metadata (
                version_key VARCHAR PRIMARY KEY,
                node_name VARCHAR NOT NULL,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                input_signature VARCHAR,  -- JSON representation of input df names and parameters
                metadata VARCHAR  -- Additional metadata as JSON
            )
        """)
        
        # Create a table to track results for each version
        self.conn.execute("""
            CREATE TABLE version_results (
                result_id VARCHAR PRIMARY KEY,
                version_key VARCHAR NOT NULL,
                output_name VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL,  -- Name of the table where the actual data is stored
                row_count INTEGER,
                created_at TIMESTAMP,
                FOREIGN KEY (version_key) REFERENCES version_metadata(version_key)
            )
        """)
        
        self.conn.commit()
    
    def store_result(self, 
                    version_key: str, 
                    node_name: str, 
                    output_name: str, 
                    df: pd.DataFrame,
                    input_signature: Dict[str, Any] = None,
                    metadata: Dict[str, Any] = None) -> bool:
        """
        Store a dataframe result with version information.
        
        Args:
            version_key: The unique version key for this result
            node_name: The name of the node that produced this result
            output_name: The name given to this dataframe output
            df: The pandas DataFrame to store
            input_signature: Optional information about inputs used to generate this output
            metadata: Optional additional metadata about this result
        
        Returns:
            bool: True if successful, False otherwise
        """
        result_id = f"{version_key}_{output_name}"
        # Make table name safe for SQL
        safe_name = f"ver_{version_key[:8]}_{node_name}_{output_name}"
        safe_name = ''.join(char if char.isalnum() else '_' for char in safe_name)
        
        # Add version metadata if it doesn't exist
        try:
            # Get current timestamp
            import datetime
            current_time = datetime.datetime.now()
            
            # Check if version exists
            exists = self.conn.execute(
                "SELECT COUNT(*) FROM version_metadata WHERE version_key = ?",
                (version_key,)
            ).fetchone()[0]
            
            if exists:
                # Update existing version
                self.conn.execute(
                    """
                    UPDATE version_metadata 
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE version_key = ?
                    """,
                    (current_time, version_key)
                )
            else:
                # Insert new version
                self.conn.execute(
                    """
                    INSERT INTO version_metadata (
                        version_key, node_name, input_signature, metadata, 
                        created_at, last_accessed
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_key, 
                        node_name,
                        json.dumps(input_signature) if input_signature else None,
                        json.dumps(metadata) if metadata else None,
                        current_time,
                        current_time
                    )
                )
            
            # Check if DataFrame is empty
            row_count = 0
            if df is not None:
                row_count = len(df)
                
            if df is None or df.empty:
                # Create an empty table with metadata
                self.conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
                columns = "dummy INTEGER" if df is None else ", ".join(f'"{col}" VARCHAR' for col in df.columns)
                self.conn.execute(f"CREATE TABLE {safe_name} ({columns})")
            else:
                # Register the dataframe with DuckDB
                self.conn.register("temp_df", df)
                
                # Create the table from the registered dataframe
                self.conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
                self.conn.execute(f"CREATE TABLE {safe_name} AS SELECT * FROM temp_df")
            
            # Add result metadata
            current_time = datetime.datetime.now()
            
            # Check if result exists
            exists = self.conn.execute(
                "SELECT COUNT(*) FROM version_results WHERE result_id = ?", 
                (result_id,)
            ).fetchone()[0]
            
            if exists:
                # Update existing result
                self.conn.execute(
                    """
                    UPDATE version_results 
                    SET table_name = ?,
                        row_count = ?,
                        created_at = ?
                    WHERE result_id = ?
                    """,
                    (safe_name, row_count, current_time, result_id)
                )
            else:
                # Insert new result
                self.conn.execute(
                    """
                    INSERT INTO version_results 
                    (result_id, version_key, output_name, table_name, row_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (result_id, version_key, output_name, safe_name, row_count, current_time)
                )
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error storing versioned result: {e}")
            try:
                self.conn.rollback()
            except:
                pass
            return False

    def store_results(self, 
                     version_key: str, 
                     node_name: str, 
                     results: Dict[str, pd.DataFrame],
                     input_signature: Dict[str, Any] = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Store multiple dataframe results with the same version information.
        
        Args:
            version_key: The unique version key for these results
            node_name: The name of the node that produced these results
            results: Dictionary of dataframes to store (name -> DataFrame)
            input_signature: Optional information about inputs used to generate these outputs
            metadata: Optional additional metadata about these results
        
        Returns:
            bool: True if all results were stored successfully
        """
        success = True
        for output_name, df in results.items():
            result_success = self.store_result(
                version_key, node_name, output_name, df, 
                input_signature, metadata
            )
            success = success and result_success
        
        return success
    
    def get_result(self, version_key: str, output_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached result by version key and output name.
        
        Args:
            version_key: The version key to look up
            output_name: The name of the output to retrieve
        
        Returns:
            Optional[pd.DataFrame]: The cached dataframe, or None if not found
        """
        try:
            # Check if this version exists
            result = self.conn.execute(
                """
                SELECT vr.table_name, vm.version_key
                FROM version_results vr
                JOIN version_metadata vm ON vr.version_key = vm.version_key
                WHERE vr.version_key = ? AND vr.output_name = ?
                """,
                (version_key, output_name)
            ).fetchone()
            
            if not result:
                return None
            
            table_name = result[0]
            
            # Get current timestamp
            import datetime
            current_time = datetime.datetime.now()
            
            # Update access time and count
            self.conn.execute(
                """
                UPDATE version_metadata
                SET last_accessed = ?, access_count = access_count + 1
                WHERE version_key = ?
                """,
                (current_time, version_key)
            )
            
            # Check if table exists
            table_exists = self.conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            
            if table_exists:
                # Fetch the dataframe
                df = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                self.conn.commit()
                return df
            else:
                print(f"Warning: Table {table_name} not found for version {version_key}")
                return None
        
        except Exception as e:
            print(f"Error retrieving versioned result: {e}")
            return None
    
    def get_results(self, version_key: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all cached results for a version key.
        
        Args:
            version_key: The version key to look up
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of output names to cached dataframes
        """
        results = {}
        
        try:
            # Find all results for this version
            output_names = self.conn.execute(
                "SELECT output_name FROM version_results WHERE version_key = ?",
                (version_key,)
            ).fetchall()
            
            for row in output_names:
                output_name = row[0]
                df = self.get_result(version_key, output_name)
                if df is not None:
                    results[output_name] = df
            
            return results
        
        except Exception as e:
            print(f"Error retrieving versioned results: {e}")
            return {}
    
    def find_similar_version(self, 
                           node_name: str, 
                           input_signature: Dict[str, Any],
                           max_age_days: int = 30) -> Optional[str]:
        """
        Find a version with similar inputs to potentially reuse cached results.
        
        Args:
            node_name: The name of the node
            input_signature: Information about inputs used
            max_age_days: Maximum age in days for cached versions to consider
        
        Returns:
            Optional[str]: A version key for a similar version, or None if none found
        """
        try:
            # Get recent versions for this node
            min_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            versions = self.conn.execute(
                """
                SELECT version_key, input_signature
                FROM version_metadata
                WHERE node_name = ? AND created_at >= ?
                ORDER BY last_accessed DESC
                """,
                (node_name, min_date)
            ).fetchall()
            
            # Simple approach: check JSON equality of input_signature
            # A more sophisticated approach would parse and compare the signatures
            input_sig_str = json.dumps(input_signature, sort_keys=True)
            
            for version_row in versions:
                version_key = version_row[0]
                stored_sig_str = version_row[1]
                
                if stored_sig_str and input_sig_str == stored_sig_str:
                    return version_key
            
            return None
        
        except Exception as e:
            print(f"Error finding similar version: {e}")
            return None
    
    def delete_version(self, version_key: str) -> bool:
        """
        Delete a version and all its associated results.
        
        Args:
            version_key: The version key to delete
        
        Returns:
            bool: True if successful
        """
        try:
            # Find all tables for this version
            tables = self.conn.execute(
                "SELECT table_name FROM version_results WHERE version_key = ?",
                (version_key,)
            ).fetchall()
            
            # Drop each table
            for row in tables:
                table_name = row[0]
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Delete metadata records
            self.conn.execute(
                "DELETE FROM version_results WHERE version_key = ?", 
                (version_key,)
            )
            self.conn.execute(
                "DELETE FROM version_metadata WHERE version_key = ?", 
                (version_key,)
            )
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error deleting version: {e}")
            try:
                self.conn.rollback()
            except:
                pass
            return False
    
    def cleanup_old_versions(self, max_age_days: int = 90, min_access_count: int = 5) -> int:
        """
        Delete old versions that haven't been accessed frequently.
        
        Args:
            max_age_days: Maximum age in days for versions to keep
            min_access_count: Minimum number of accesses for versions to keep
                             regardless of age
        
        Returns:
            int: Number of versions deleted
        """
        try:
            # Find versions to delete
            min_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            versions_to_delete = self.conn.execute(
                """
                SELECT version_key
                FROM version_metadata
                WHERE (last_accessed < ? AND access_count < ?)
                """,
                (min_date, min_access_count)
            ).fetchall()
            
            # Delete each version
            deleted_count = 0
            for row in versions_to_delete:
                version_key = row[0]
                success = self.delete_version(version_key)
                if success:
                    deleted_count += 1
            
            return deleted_count
        
        except Exception as e:
            print(f"Error during version cleanup: {e}")
            return 0
    
    def version_exists(self, version_key: str) -> bool:
        """
        Check if a version exists in the cache.
        
        Args:
            version_key: The version key to check
        
        Returns:
            bool: True if the version exists
        """
        try:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM version_metadata WHERE version_key = ?",
                (version_key,)
            ).fetchone()[0]
            
            return count > 0
        except Exception:
            return False
    
    def get_version_info(self, version_key: str) -> Dict[str, Any]:
        """
        Get metadata about a specific version.
        
        Args:
            version_key: The version key to query
        
        Returns:
            Dict[str, Any]: Dictionary of version metadata
        """
        try:
            result = self.conn.execute(
                """
                SELECT 
                    vm.node_name, 
                    vm.created_at, 
                    vm.last_accessed, 
                    vm.access_count, 
                    vm.input_signature, 
                    vm.metadata,
                    COUNT(vr.result_id) as result_count
                FROM version_metadata vm
                LEFT JOIN version_results vr ON vm.version_key = vr.version_key
                WHERE vm.version_key = ?
                GROUP BY vm.version_key
                """,
                (version_key,)
            ).fetchone()
            
            if not result:
                return {}
            
            return {
                'version_key': version_key,
                'node_name': result[0],
                'created_at': result[1],
                'last_accessed': result[2],
                'access_count': result[3],
                'input_signature': json.loads(result[4]) if result[4] else None,
                'metadata': json.loads(result[5]) if result[5] else None,
                'result_count': result[6]
            }
        
        except Exception as e:
            print(f"Error getting version info: {e}")
            return {}
    
    def list_versions(self, 
                     node_name: str = None, 
                     limit: int = 100,
                     order_by: str = 'last_accessed',
                     order_dir: str = 'DESC') -> List[Dict[str, Any]]:
        """
        List versions in the cache, with optional filtering.
        
        Args:
            node_name: Optional node name to filter by
            limit: Maximum number of versions to return
            order_by: Field to order results by
            order_dir: Direction to order results (ASC or DESC)
        
        Returns:
            List[Dict[str, Any]]: List of version metadata dictionaries
        """
        try:
            # Validate order_by parameter to prevent SQL injection
            valid_order_fields = ['created_at', 'last_accessed', 'access_count']
            if order_by not in valid_order_fields:
                order_by = 'last_accessed'
            
            # Validate order_dir parameter
            if order_dir not in ['ASC', 'DESC']:
                order_dir = 'DESC'
            
            where_clause = "WHERE 1=1"
            params = []
            
            if node_name:
                where_clause += " AND vm.node_name = ?"
                params.append(node_name)
            
            query = f"""
                SELECT 
                    vm.version_key,
                    vm.node_name, 
                    vm.created_at, 
                    vm.last_accessed, 
                    vm.access_count,
                    COUNT(vr.result_id) as result_count
                FROM version_metadata vm
                LEFT JOIN version_results vr ON vm.version_key = vr.version_key
                {where_clause}
                GROUP BY vm.version_key
                ORDER BY vm.{order_by} {order_dir}
                LIMIT ?
            """
            params.append(limit)
            
            results = self.conn.execute(query, params).fetchall()
            
            versions = []
            for row in results:
                versions.append({
                    'version_key': row[0],
                    'node_name': row[1],
                    'created_at': row[2],
                    'last_accessed': row[3],
                    'access_count': row[4],
                    'result_count': row[5]
                })
            
            return versions
        
        except Exception as e:
            print(f"Error listing versions: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

# Singleton for global access to the version cache
_version_cache_instance = None

def get_version_cache(db_path: str = None) -> DuckDBVersionCache:
    """
    Get the singleton instance of DuckDBVersionCache.
    
    Args:
        db_path: Optional path to the DuckDB database file
    
    Returns:
        DuckDBVersionCache: The singleton instance
    """
    global _version_cache_instance
    if _version_cache_instance is None:
        _version_cache_instance = DuckDBVersionCache(db_path)
    return _version_cache_instance
