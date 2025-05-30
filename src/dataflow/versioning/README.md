# Dataflow Versioning System: Tutorial and Usage Guide

## Introduction

The Dataflow Versioning System provides an intelligent caching mechanism for dataflow operations. It tracks input dataframes, processing functions, and parameters to avoid redundant computation when the same operation is requested multiple times. This guide explains how to use the versioning system and provides best practices for integrating it into your workflow.

## Key Benefits

- **Avoid Redundant Computation**: Skip calculations that have been performed before with identical inputs and parameters
- **Improved Performance**: Significant speedup for repeated operations, especially for complex processing
- **Memory Efficiency**: Reduced memory usage during execution with cached results
- **Support for ML Models**: Can cache machine learning models between training and prediction
- **Automatic Detection**: Intelligently determines when cached results can be reused

## Core Components

The versioning system consists of three main components:

1. **Version Utilities** (`version_utils.py`): Functions for hashing dataframes and functions, generating version keys, and comparing results.

2. **Version Cache** (`version_cache.py`): DuckDB-based cache for storing and retrieving versioned results.

3. **Versioned Execution** (`versioned_execution.py`): Execution nodes and dataflows that integrate with the versioning system.

## Basic Usage

### 1. Creating a Versioned Dataflow

To use versioning, create a dataflow with `VersionedDuckDBDataflow` and `VersionedDuckDBExecutionNode`:

```python
# Get manager instance
manager = ParallelExecutionDataFlowManager.getInstance()

# Create a dataflow
base_dataflow = manager.newDataFlow(NodeClass)
dataflow = VersionedDuckDBDataflow(base_dataflow)

# Create versioned nodes
node_a = base_dataflow.node("NodeA", VersionedDuckDBExecutionNode)
node_a.process_func = process_func_a

node_b = base_dataflow.node("NodeB", VersionedDuckDBExecutionNode)
node_b.process_func = process_func_b
node_b.add_dependency(node_a)
```

### 2. Controlling Caching Behavior

You can control which functions have their results cached using the `@cacheable` decorator:

```python
# Function with results that should be cached (default)
@cacheable
def stable_function(dfs, params):
    # Results will be cached based on inputs and parameters
    return {'output': processed_data}

# Function with results that should NOT be cached
@cacheable(enabled=False)
def dynamic_function(dfs, params):
    # Results will always be recomputed
    return {'output': processed_data}
```

### 3. Executing the Dataflow

Execute the dataflow normally. The versioning system automatically handles caching:

```python
# First execution (will compute and cache)
results = dataflow.execute(parameters)

# Second execution with same parameters (will use cache)
results = dataflow.execute(parameters)

# Execution with different parameters (will compute)
results = dataflow.execute(different_parameters)
```

## Advanced Usage

### Working with Machine Learning Models

The versioning system can cache machine learning models, separating the training and prediction phases:

```python
# Define a training function that returns a trained model
@cacheable
def train_model(dfs, params):
    # Get training data
    training_data = dfs.get('training_data')
    
    # Create and train model
    model = create_model(params)
    model.fit(training_data)
    
    # Return the trained model (will be cached)
    return {'model': model}

# Define a prediction function that uses the trained model
def predict(dfs, params):
    # Get the trained model from cache
    model = dfs.get('model')
    data = dfs.get('input_data')
    
    # Make predictions
    predictions = model.predict(data)
    
    return {'predictions': predictions}
```

### Versioning with Time Series Data

For time series data, versioning can cache results for specific date ranges:

```python
@cacheable
def process_time_series(dfs, params):
    # Get start and end dates
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    
    # Filter data by date range and process
    # ...
    
    return {'processed': result_df}
```

### Disabling Versioning for Specific Nodes

You can disable caching for specific nodes:

```python
# Create a node with versioning disabled
node = base_dataflow.node("DynamicNode", VersionedDuckDBExecutionNode)
node.process_func = dynamic_process_func
node.set_cache_enabled(False)  # Disable caching for this node
```

### Disabling Versioning for the Entire Dataflow

You can disable versioning for the entire dataflow:

```python
# Disable versioning for all nodes in the dataflow
dataflow.set_versioning_enabled(False)
```

## How Versioning Works

1. Each execution node looks at:
   - The processing function's source code
   - The input dataframes' content
   - The parameters passed to the function

2. A unique version key is generated from these components.

3. Before executing, the system checks if results for this version key already exist.

4. If found, the cached results are used; otherwise, the function is executed and results are cached.

## Best Practices

1. **Use the `@cacheable` Decorator**: Mark functions based on whether their results should be cached.

2. **Control Cache Size**: Periodically clean up old versions using `cleanup_old_versions()`.

3. **Function Purity**: For optimal caching, make processing functions depend only on their inputs and parameters.

4. **Parameter Consistency**: Pass all relevant parameters explicitly rather than using global state.

5. **Memory Management**: Clear execution data when no longer needed with `dataflow.clear_execution_data()`.

## Common Patterns

### 1. ETL Pipelines

Cache expensive data transformations:

```python
@cacheable
def transform_data(dfs, params):
    # Expensive transformations here
    return {'transformed': result}
```

### 2. Report Generation

Cache report data but not the rendering:

```python
@cacheable
def calculate_metrics(dfs, params):
    # Complex calculations
    return {'metrics': results}

@cacheable(enabled=False)
def render_report(dfs, params):
    # Generate report (don't cache)
    return {'report': report_content}
```

### 3. Incremental Processing

Process only new data while reusing previous results:

```python
@cacheable
def process_batch(dfs, params):
    batch_id = params.get('batch_id')
    # Process specific batch
    return {f'batch_{batch_id}': result}
```

## Troubleshooting

1. **Unexpected Cache Misses**: Check if functions are deterministic and all relevant parameters are included.

2. **Performance Issues**: Ensure large dataframes are handled efficiently; use sampling for hashing.

3. **Memory Problems**: Clear old execution data and periodically clean up the cache.

4. **Non-Serializable Objects**: Some objects might not cache properly; ensure they can be pickled.

## Examples

See `/home/sobibence/AAU/3_semester/project/psa_tool/examples/versioning_examples.py` for complete working examples.
