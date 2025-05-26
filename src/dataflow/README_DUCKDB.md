# DuckDB Integration for Parallel Dataflow

This document explains how DuckDB is used for persistence in the parallel dataflow framework.

## Overview

The parallel dataflow framework has been enhanced with DuckDB integration to:

1. **Reduce memory consumption** - Large dataframes are stored in DuckDB instead of kept in memory
2. **Enable persistence** - Results survive application restarts
3. **Improve performance** - DuckDB is fast for analytical workloads
4. **Support SQL queries** - Results can be accessed with SQL

## Architecture

The integration consists of:

1. `DuckDBDataflowManager` - Manages the DuckDB database and provides methods to store/retrieve dataframes
2. `DuckDBParallelExecutionNode` - Extends `ParallelExecutionNode` to use DuckDB for data exchange
3. `DuckDBParallelDataflow` - Wraps a `ParallelExecutionDataflow` to add DuckDB persistence

## Database Schema

The database has the following tables:

1. `dataflow_executions` - Tracks each execution of a dataflow
   - `execution_id` - Unique identifier
   - `dataflow_name` - Name of the dataflow class
   - `start_time` - When execution started
   - `end_time` - When execution completed
   - `status` - Status of execution (running, completed, error)
   - `parameters` - Serialized parameters

2. `node_results` - Tracks results from each node
   - `result_id` - Unique identifier for the result
   - `execution_id` - Links to the execution
   - `node_name` - Name of the node
   - `table_name` - Name of the table containing the data
   - `created_at` - When the result was stored

3. Dynamic tables for each dataframe result, named:
   - `df_<execution_id>_<node_name>_<dataframe_name>`

## Usage Example

```python
# Create a DuckDB-backed dataflow
from src.dataflow.parallel_execution_dataflow import ParallelExecutionDataFlowManager
from src.dataflow.duckdb_parallel_execution import DuckDBParallelExecutionNode, DuckDBParallelDataflow

# Get the manager and create a dataflow
manager = ParallelExecutionDataFlowManager.getInstance()
dataflow = manager.newDataFlow(MyNodeClass)

# Wrap with DuckDB persistence
duckdb_dataflow = DuckDBParallelDataflow(dataflow)

# Create nodes with DuckDB persistence
node_a = dataflow.node("A", DuckDBParallelExecutionNode)
node_a.process_func = my_processing_function

# Execute the dataflow
results = duckdb_dataflow.execute(parameters)

# You can retrieve results from DuckDB afterwards
from src.dataflow.duckdb_dataflow_utils import DuckDBDataflowManager
db_manager = DuckDBDataflowManager.get_instance()
node_results = db_manager.load_node_results(duckdb_dataflow._execution_id, "A")
```

## Benefits

1. **Memory Efficiency** - Dataframes are offloaded to disk, freeing up memory
2. **Persistence** - Results are stored permanently until explicitly cleared
3. **Scalability** - Can handle larger datasets than in-memory processing
4. **SQL Queries** - Results can be analyzed with SQL queries directly
5. **Resilience** - Processing can be resumed after crashes

## Limitations

1. **Performance Overhead** - Some overhead for storing/retrieving data
2. **Storage Space** - Requires disk space for large dataframes
3. **Data Types** - Limited to data types supported by DuckDB

## Future Enhancements

1. **Automatic Cleanup** - Purge old executions automatically
2. **Compression Options** - Add support for compressed storage
3. **Result Expiry** - Set TTL for execution results
4. **Column Filtering** - Only load required columns
5. **Query Pushdown** - Push filters and aggregations to DuckDB
