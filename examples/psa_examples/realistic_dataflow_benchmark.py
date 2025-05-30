#!/usr/bin/env python3
"""
Realistic Parallel vs Serial Dataflow Benchmark

This example demonstrates the performance difference between:
1. Serial execution using standard DataProcessingNode
2. Parallel execution using ParallelExecutionNode
3. Parallel dataflow execution using ParallelDataFlowManager
4. Combined parallel execution using both techniques

It measures and compares the execution times for real-world data processing tasks
including data loading, merging, transformation, feature engineering, and 
machine learning prediction on energy price data.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import multiprocessing
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add project root to Python path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_root)

# Import from src or directly depending on the setup
try:
    from src.dataflow.parallel_execution_dataflow import ParallelExecutionNode
    from src.dataflow.dataflow_manager_paralell import ParallelDataFlowManager
    from src.dataflow.dataflow_classes import DataProcessingNode
    from src.optimization.solver_classes import Node
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    try:
        from dataflow.parallel_execution_dataflow import ParallelExecutionNode
        from dataflow.dataflow_manager_paralell import ParallelDataFlowManager
        from dataflow.dataflow_classes import DataProcessingNode
        from optimization.solver_classes import Node
    except ImportError as e:
        print(f"Alternative import also failed: {e}")
        sys.exit(1)

print(f"Using project root: {project_root}")

# Define data paths
DATA_DIR = os.path.join(project_root, "data")
SPOT_PRICES_PATH = os.path.join(DATA_DIR, "spot_prices_test.csv")
FUTURE_PRICES_PATH = os.path.join(DATA_DIR, "future_prices.csv")

# Define a node for the parallel dataflow
class BenchmarkNode(Node):
    def __init__(self, name):
        super().__init__(name)
    
    def set_time_length(self, time_len):
        pass
    
    def constraints(self, t):
        return []
    
    @property
    def cost(self):
        return 0

# Create a simple dataflow for serial execution
class SimpleDataflow:
    def __init__(self):
        self.nodes = {}
        self.dependencies = {}
        self.final_node = None
    
    def add_node(self, node, dependencies=None, final=False):
        if dependencies is None:
            dependencies = []
            
        self.nodes[node.name] = node
        self.dependencies[node.name] = dependencies
        
        if final:
            self.final_node = node
    
    def execute(self, dfs, parameters):
        # Simple topological sort execution
        processed_nodes = {}  # Store the actual result dictionaries
        
        def process_node(node_name):
            if node_name in processed_nodes:
                return processed_nodes[node_name]
            
            node = self.nodes[node_name]
            deps = self.dependencies[node_name]
            
            # Process dependencies first
            dep_results = {}
            for dep in deps:
                if dep in self.nodes:
                    dep_results[dep] = process_node(dep)
            
            # Execute this node with the actual dependency results
            result = node.process_func(dep_results, parameters)
            processed_nodes[node_name] = result
            return result
        
        # Process the final node which will recursively process all dependencies
        if self.final_node:
            final_result = process_node(self.final_node.name)
            return final_result
        return None

# Data Loading Functions
def load_data_processing(dfs, parameters):
    """Load energy price data based on parameters"""
    data_path = parameters.get('data_path', SPOT_PRICES_PATH)
    print(f"Loading data from {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        if 'sample_size' in parameters:
            data = data.sample(min(parameters['sample_size'], len(data)))
        return {'data': data}
    except Exception as e:
        print(f"Error loading data: {e}")
        return {'data': pd.DataFrame()}

# Data Transformation Functions
def transform_data_processing(dfs, parameters):
    """Transform the loaded data"""
    # More flexible input handling for both serial and parallel execution
    data = None
    
    if not dfs:
        print("No inputs provided for transformation")
        return {'transformed_data': pd.DataFrame()}
    
    # Print keys for debugging
    print(f"Transform input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    # Handle different possible input structures:
    try:
        # Case 1: Serial format - nested dict structure
        if 'load_data' in dfs:
            if isinstance(dfs['load_data'], dict) and 'data' in dfs['load_data']:
                data = dfs['load_data']['data'].copy()
                print("Found data in dfs['load_data']['data']")
            elif isinstance(dfs['load_data'], pd.DataFrame):
                data = dfs['load_data'].copy()
                print("Found DataFrame directly in dfs['load_data']")
        
        # Case 2: Direct access to 'data' (common in parallel mode)
        elif 'data' in dfs:
            if isinstance(dfs['data'], pd.DataFrame):
                data = dfs['data'].copy()
                print("Found DataFrame directly in dfs['data']")
            else:
                print(f"dfs['data'] exists but is not a DataFrame: {type(dfs['data'])}")
                
        # Case 3: Try to find any DataFrame in the dict (last resort)
        else:
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame):
                    data = value.copy()
                    print(f"Found DataFrame at key '{key}'")
                    break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame):
                            data = sub_value.copy()
                            print(f"Found DataFrame at key '{key}.{sub_key}'")
                            break
    except Exception as e:
        print(f"Error accessing data in transform: {e}")
        import traceback
        traceback.print_exc()
            
    if data is None:
        print("No input data found for transformation")
        return {'transformed_data': pd.DataFrame()}
    
    print(f"Transforming data with {len(data)} rows")
    
    try:
        # Convert time columns to datetime
        data['HourUTC'] = pd.to_datetime(data['HourUTC'])
        data['HourDK'] = pd.to_datetime(data['HourDK'])
        
        # Extract time features
        data['Hour'] = data['HourDK'].dt.hour
        data['Day'] = data['HourDK'].dt.day
        data['Month'] = data['HourDK'].dt.month
        data['Year'] = data['HourDK'].dt.year
        data['DayOfWeek'] = data['HourDK'].dt.dayofweek
        data['Weekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Simulate a computationally intensive operation
        intensity = parameters.get('transform_intensity', 1)
        for _ in range(intensity):
            # Calculate rolling statistics (intensive operation)
            window_size = 24  # 24-hour window
            data['Price_Mean_24h'] = data.groupby('PriceArea')['SpotPriceDKK'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
            data['Price_Std_24h'] = data.groupby('PriceArea')['SpotPriceDKK'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).std()
            )
            data['Price_Min_24h'] = data.groupby('PriceArea')['SpotPriceDKK'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).min()
            )
            data['Price_Max_24h'] = data.groupby('PriceArea')['SpotPriceDKK'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).max()
            )
        
        return {'transformed_data': data}
    except Exception as e:
        print(f"Error transforming data: {e}")
        return {'transformed_data': pd.DataFrame()}

# Feature Engineering Functions
def engineer_features_processing(dfs, parameters):
    """Engineer additional features from the transformed data"""
    # More flexible input handling for both serial and parallel execution
    data = None
    
    if not dfs:
        print("No inputs provided for feature engineering")
        return {'engineered_data': pd.DataFrame()}
    
    # Print keys for debugging
    print(f"Feature engineering input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    try:
        # Case 1: Serial format with nested dict
        if 'transform_data' in dfs:
            if isinstance(dfs['transform_data'], dict) and 'transformed_data' in dfs['transform_data']:
                data = dfs['transform_data']['transformed_data'].copy()
                print("Found data in dfs['transform_data']['transformed_data']")
            elif isinstance(dfs['transform_data'], pd.DataFrame):
                data = dfs['transform_data'].copy()
                print("Found DataFrame directly in dfs['transform_data']")
                
        # Case 2: Direct access to 'transformed_data' (common in parallel mode)
        elif 'transformed_data' in dfs:
            if isinstance(dfs['transformed_data'], pd.DataFrame):
                data = dfs['transformed_data'].copy()
                print("Found DataFrame directly in dfs['transformed_data']")
                
        # Case 3: Look for any DataFrame with the right columns
        else:
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame):
                    # Check if it has at least some expected columns
                    if 'SpotPriceDKK' in value.columns:
                        data = value.copy()
                        print(f"Found suitable DataFrame at key '{key}'")
                        break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame) and 'SpotPriceDKK' in sub_value.columns:
                            data = sub_value.copy()
                            print(f"Found suitable DataFrame at key '{key}.{sub_key}'")
                            break
    except Exception as e:
        print(f"Error accessing data in feature engineering: {e}")
        import traceback
        traceback.print_exc()
    
    if data is None:
        print("No transformed data found for feature engineering")
        return {'engineered_data': pd.DataFrame()}
    
    print(f"Engineering features for {len(data)} rows")
    
    try:
        # Create lag features (previous hours' prices)
        for i in range(1, 25):
            data[f'Price_Lag_{i}h'] = data.groupby('PriceArea')['SpotPriceDKK'].shift(i)
        
        # Calculate price differences
        data['Price_Diff_1h'] = data['SpotPriceDKK'] - data['Price_Lag_1h']
        data['Price_Ratio_1h'] = data['SpotPriceDKK'] / data['Price_Lag_1h'].replace(0, np.nan)
        
        # Time-based cyclical features
        data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
        data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
        data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Simulate a computationally intensive operation
        intensity = parameters.get('feature_intensity', 1)
        for _ in range(intensity):
            # Calculate exponentially weighted features
            for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                data[f'EWM_{alpha}'] = data.groupby('PriceArea')['SpotPriceDKK'].transform(
                    lambda x: x.ewm(alpha=alpha).mean()
                )
        
        return {'engineered_data': data}
    except Exception as e:
        print(f"Error engineering features: {e}")
        import traceback
        traceback.print_exc()
        return {'engineered_data': pd.DataFrame()}

# Machine Learning Model Training
def train_model_processing(dfs, parameters):
    """Train a machine learning model on the engineered data"""
    # Handle various input formats
    data = None
    
    # Print keys for debugging
    print(f"Training model input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    try:
        # Case 1: Serial format with nested dict
        if dfs and 'engineer_features' in dfs:
            if isinstance(dfs['engineer_features'], dict) and 'engineered_data' in dfs['engineer_features']:
                data = dfs['engineer_features']['engineered_data']
                print("Found data in dfs['engineer_features']['engineered_data']")
            elif isinstance(dfs['engineer_features'], pd.DataFrame):
                data = dfs['engineer_features']
                print("Found DataFrame directly in dfs['engineer_features']")
                
        # Case 2: Direct access to 'engineered_data' (common in parallel mode)
        elif dfs and 'engineered_data' in dfs:
            if isinstance(dfs['engineered_data'], pd.DataFrame):
                data = dfs['engineered_data']
                print("Found DataFrame directly in dfs['engineered_data']")
                
        # Case 3: Look for any DataFrame with expected columns
        elif dfs:
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame) and 'SpotPriceDKK' in value.columns:
                    data = value
                    print(f"Found suitable DataFrame at key '{key}'")
                    break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame) and 'SpotPriceDKK' in sub_value.columns:
                            data = sub_value
                            print(f"Found suitable DataFrame at key '{key}.{sub_key}'")
                            break
    except Exception as e:
        print(f"Error accessing data for model training: {e}")
        import traceback
        traceback.print_exc()
    
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        print("No engineered data found for model training")
        return {'model': None, 'scaler': None, 'features': None, 'X_test': None, 'y_test': None}
    
    print(f"Training model on {len(data)} rows")
    
    try:
        # Define features and target
        target_col = 'SpotPriceDKK'
        exclude_cols = ['HourUTC', 'HourDK', 'SpotPriceDKK', 'SpotPriceEUR', 'PriceArea']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare data
        X = data[feature_cols]
        y = data[target_col]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model_complexity = parameters.get('model_complexity', 'simple')
        if model_complexity == 'complex':
            # More complex model with more trees
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Force single core to better measure parallelization effects
            )
        else:
            # Simpler model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=1  # Force single core to better measure parallelization effects
            )
        
        # Train the model (this is CPU intensive)
        model.fit(X_train_scaled, y_train)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': feature_cols,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
    except Exception as e:
        print(f"Error training model: {e}")
        return {'model': None, 'scaler': None, 'features': None, 'X_test': None, 'y_test': None}

# Prediction and Evaluation
def predict_and_evaluate_processing(dfs, parameters):
    """Make predictions and evaluate the model"""
    # Handle various input formats
    # Print keys for debugging
    print(f"Prediction input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    model = None
    X_test = None
    y_test = None
    
    try:
        # Case 1: Serial format - nested dict structure
        if dfs and 'train_model' in dfs:
            model_data = dfs['train_model']
            if isinstance(model_data, dict):
                model = model_data.get('model')
                X_test = model_data.get('X_test')
                y_test = model_data.get('y_test')
                print("Found model data in dfs['train_model'] dictionary")
                
        # Case 2: Direct access to model components (parallel mode may flatten things)
        elif dfs:
            if 'model' in dfs:
                model = dfs['model']
                print("Found model directly in dfs['model']")
                
            if 'X_test' in dfs:
                X_test = dfs['X_test']
                print("Found X_test directly in dfs['X_test']")
                
            if 'y_test' in dfs:
                y_test = dfs['y_test']
                print("Found y_test directly in dfs['y_test']")
    except Exception as e:
        print(f"Error accessing model data for prediction: {e}")
        import traceback
        traceback.print_exc()
    
    if not model or X_test is None or y_test is None:
        print("Model data is incomplete")
        return {'predictions': None, 'metrics': {}}
    
    print("Making predictions and evaluating model")
    
    try:
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # Feature importance - more robust access to features
        features = None
        if dfs and 'train_model' in dfs and isinstance(dfs['train_model'], dict) and 'features' in dfs['train_model']:
            features = dfs['train_model']['features']
        elif dfs and 'features' in dfs:
            features = dfs['features']
        
        # If we have features, create the feature importance dict
        if features is not None:
            # Make sure we have the right number of features
            if len(features) == len(model.feature_importances_):
                feature_importances = dict(zip(
                    features,
                    model.feature_importances_
                ))
            else:
                # Handle mismatch in feature count
                print(f"Warning: Feature count mismatch ({len(features)} names vs {len(model.feature_importances_)} importances)")
                feature_importances = {f"Feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
            
            # Sort feature importances
            sorted_importances = dict(sorted(
                feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            )[:10])  # Top 10 features
        else:
            # If no features available, provide feature importance with generic names
            feature_importances = {f"Feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
            sorted_importances = dict(sorted(
                feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            )[:10])
            print("Using generic feature names for importance as feature names were not available")
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'feature_importances': sorted_importances
        };
        
        return {'predictions': predictions, 'metrics': metrics}
    except Exception as e:
        print(f"Error in prediction and evaluation: {e}")
        return {'predictions': None, 'metrics': {}}

# Result Aggregation
def aggregate_results_processing(dfs, parameters):
    """Aggregate all results into a final output"""
    print("Aggregating final results")
    
    # Print keys for debugging
    print(f"Aggregation input keys: {list(dfs.keys()) if dfs else 'None'}")
    
    try:
        # Get metrics, handling both serial and parallel formats
        metrics = {}
        
        # Case 1: Metrics from predict_evaluate node
        if dfs and 'predict_evaluate' in dfs:
            if isinstance(dfs['predict_evaluate'], dict):
                if 'metrics' in dfs['predict_evaluate']:
                    metrics = dfs['predict_evaluate']['metrics']
                    print("Found metrics in dfs['predict_evaluate']['metrics']")
            elif hasattr(dfs['predict_evaluate'], 'get'):
                metrics = dfs['predict_evaluate'].get('metrics', {})
                print("Found metrics using get() on dfs['predict_evaluate']")
                
        # Case 2: Direct access to metrics (common in parallel mode)
        elif dfs and 'metrics' in dfs:
            metrics = dfs['metrics']
            print("Found metrics directly in dfs['metrics']")
        
        # Get data size, handling both serial and parallel formats
        data_size = 0
        
        # Case 1: Data size from engineered_data
        if dfs and 'engineer_features' in dfs:
            eng_data = None
            if isinstance(dfs['engineer_features'], dict) and 'engineered_data' in dfs['engineer_features']:
                eng_data = dfs['engineer_features']['engineered_data']
                print("Found engineered data in nested dict")
            elif isinstance(dfs['engineer_features'], pd.DataFrame):
                eng_data = dfs['engineer_features']
                print("Found engineered data directly as DataFrame")
                
            if isinstance(eng_data, pd.DataFrame):
                data_size = len(eng_data)
                
        # Case 2: Direct access to engineered_data
        elif dfs and 'engineered_data' in dfs:
            if isinstance(dfs['engineered_data'], pd.DataFrame):
                data_size = len(dfs['engineered_data'])
                print("Found engineered_data directly in dfs")
        
        # Create a DataFrame for results that matches the expected format
        results_dict = {
            'data_size': [data_size],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        # Add metrics as columns
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key != 'feature_importances' and not isinstance(value, dict):
                    results_dict[key] = [value]
        
        # Create DataFrame from the dictionary
        results_df = pd.DataFrame(results_dict)
        
        print(f"Results: processed {data_size} rows, RMSE: {metrics.get('rmse', 'N/A')}")
        
        # Return with the key "results" which is expected by the ParallelDataflow
        return {'results': results_df}
    except Exception as e:
        print(f"Error aggregating results: {e}")
        import traceback
        traceback.print_exc()
        # Return a valid DataFrame even on error
        return {'results': pd.DataFrame({'error': [str(e)]})}

# Serial Execution Benchmark
def run_serial_benchmark(params):
    """Run full data processing pipeline in serial"""
    print(f"\nRunning serial benchmark...")
    
    start_time = time.time()
    
    # Create serial dataflow
    dataflow = SimpleDataflow()
    
    # Add processing nodes
    load_node = DataProcessingNode('load_data', load_data_processing)
    transform_node = DataProcessingNode('transform_data', transform_data_processing)
    feature_node = DataProcessingNode('engineer_features', engineer_features_processing)
    train_node = DataProcessingNode('train_model', train_model_processing)
    predict_node = DataProcessingNode('predict_evaluate', predict_and_evaluate_processing)
    aggregate_node = DataProcessingNode('aggregate_results', aggregate_results_processing)
    
    # Set up dataflow dependencies
    dataflow.add_node(load_node)
    dataflow.add_node(transform_node, ['load_data'])
    dataflow.add_node(feature_node, ['transform_data'])
    dataflow.add_node(train_node, ['engineer_features'])
    dataflow.add_node(predict_node, ['train_model'])
    dataflow.add_node(aggregate_node, ['engineer_features', 'predict_evaluate'], final=True)
    
    # Execute
    result = dataflow.execute({
        'load_data': None,
        'transform_data': None,
        'engineer_features': None,
        'train_model': None,
        'predict_evaluate': None
    }, params)
    
    duration = time.time() - start_time
    print(f"Serial benchmark completed in {duration:.2f} seconds")
    
    # Add benchmark info to results
    result['duration'] = duration
    result['method'] = 'Serial'
    
    return result, duration

# Parallel Node Benchmark
def run_parallel_node_benchmark(params):
    """Run with parallel execution nodes but serial dataflow"""
    print(f"\nRunning parallel node benchmark...")
    
    start_time = time.time()
    
    # Create serial dataflow
    dataflow = SimpleDataflow()
    
    # Add parallel execution nodes
    load_node = ParallelExecutionNode('load_data')
    load_node.process_func = load_data_processing
    
    transform_node = ParallelExecutionNode('transform_data')
    transform_node.process_func = transform_data_processing
    
    feature_node = ParallelExecutionNode('engineer_features')
    feature_node.process_func = engineer_features_processing
    
    train_node = ParallelExecutionNode('train_model')
    train_node.process_func = train_model_processing
    
    predict_node = ParallelExecutionNode('predict_evaluate')
    predict_node.process_func = predict_and_evaluate_processing
    
    aggregate_node = ParallelExecutionNode('aggregate_results')
    aggregate_node.process_func = aggregate_results_processing
    
    # Set up dataflow dependencies
    dataflow.add_node(load_node)
    dataflow.add_node(transform_node, ['load_data'])
    dataflow.add_node(feature_node, ['transform_data'])
    dataflow.add_node(train_node, ['engineer_features'])
    dataflow.add_node(predict_node, ['train_model'])
    dataflow.add_node(aggregate_node, ['engineer_features', 'predict_evaluate'], final=True)
    
    # Execute
    result = dataflow.execute({
        'load_data': None,
        'transform_data': None,
        'engineer_features': None,
        'train_model': None,
        'predict_evaluate': None
    }, params)
    
    duration = time.time() - start_time
    print(f"Parallel node benchmark completed in {duration:.2f} seconds")
    
    # Add benchmark info to results
    result['duration'] = duration
    result['method'] = 'Parallel Node'
    
    return result, duration

# Helper functions for additional processing nodes in parallel benchmark
def data_augmentation_processing(dfs, parameters):
    """Additional data augmentation stage to increase nodes in the pipeline"""
    data = None
    print(f"Augmenting data...")
    
    # Try to find data from input nodes
    try:
        if 'transformed_data' in dfs:
            data = dfs['transformed_data'].copy()
        elif 'transform_data' in dfs and 'transformed_data' in dfs['transform_data']:
            data = dfs['transform_data']['transformed_data'].copy()
        else:
            # Search for any DataFrame
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame) and 'SpotPriceDKK' in value.columns:
                    data = value.copy()
                    break
    except:
        pass
    
    if data is None:
        return {'augmented_data': pd.DataFrame()}
    
    # Create synthetic additional rows for augmentation
    try:
        # Make a copy of original data and add small random variations
        augmented = data.copy()
        for col in augmented.select_dtypes(include=[np.number]).columns:
            noise_factor = 0.05  # 5% noise
            noise = np.random.normal(0, noise_factor * augmented[col].std(), size=len(augmented))
            augmented[col] = augmented[col] + noise
        
        # Add time offset for datetime columns
        if 'HourDK' in augmented.columns:
            augmented['HourDK'] = augmented['HourDK'] + pd.Timedelta(hours=24)
        if 'HourUTC' in augmented.columns:
            augmented['HourUTC'] = augmented['HourUTC'] + pd.Timedelta(hours=24)
            
        # Add a flag to identify augmented data
        augmented['IsAugmented'] = 1
        data['IsAugmented'] = 0
        
        # Combine original and augmented data
        combined = pd.concat([data, augmented], ignore_index=True)
        
        # Make computation intensive
        for _ in range(3):
            combined['Random_Feature'] = np.random.rand(len(combined))
        
        return {'augmented_data': combined}
    except Exception as e:
        print(f"Error in data augmentation: {e}")
        return {'augmented_data': data}  # Return original data on error

def data_filtering_processing(dfs, parameters):
    """Additional filtering step to increase nodes in the pipeline"""
    data = None
    print(f"Filtering data...")
    
    # Try to find data from input nodes
    try:
        if 'augmented_data' in dfs:
            data = dfs['augmented_data'].copy() 
        elif 'data_augmentation' in dfs and 'augmented_data' in dfs['data_augmentation']:
            data = dfs['data_augmentation']['augmented_data'].copy()
        else:
            # Search for any suitable DataFrame
            for key, value in dfs.items():
                if isinstance(value, pd.DataFrame) and 'SpotPriceDKK' in value.columns:
                    data = value.copy()
                    break
    except:
        pass
    
    if data is None:
        return {'filtered_data': pd.DataFrame()}
    
    # Apply various filters to make processing intensive
    try:
        # Filter by time of day to simulate peak hour analysis
        if 'Hour' in data.columns:
            # Keep only peak hours (8-20)
            filtered_data = data[(data['Hour'] >= 8) & (data['Hour'] <= 20)]
        else:
            filtered_data = data
        
        # Filter by weekday/weekend if available
        if 'DayOfWeek' in filtered_data.columns:
            # Add workday/weekend filter
            filtered_data = filtered_data[filtered_data['DayOfWeek'] < 5]  # Only weekdays
        
        # Make this compute intensive
        for _ in range(3):
            # Add some computationally intensive operations
            filtered_data['Intensity_Score'] = np.sin(filtered_data['SpotPriceDKK']) * np.cos(filtered_data['SpotPriceDKK'])
            
        return {'filtered_data': filtered_data}
    except Exception as e:
        print(f"Error in data filtering: {e}")
        return {'filtered_data': data}  # Return original data on error

# Parallel Dataflow Benchmark
def run_parallel_dataflow_benchmark(params):
    """Run with parallel dataflow manager"""
    print(f"\nRunning parallel dataflow benchmark...")
    
    # Get manager instance - use a fresh one
    try:
        ParallelDataFlowManager._instances = {}  # Reset the singleton instances if possible
    except:
        pass
        
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        start_time = time.time()
        
        # Create dataflow with a clean slate
        dataflow = manager.newDataFlow(BenchmarkNode)
        dataflow._final_df_name = 'results'  # This tells the manager which key to look for
        
        # Enable caching to prevent redundant processing
        try:
            dataflow._enable_caching = True
        except:
            print("Caching not supported in this version")
            
        # Configure the manager for optimal performance - use more workers for parallel benefit
        manager.config_max_workers = multiprocessing.cpu_count()  # Use all cores
        manager.enable_debug = False  # Turn off debug for performance
        
        # Add nodes with clearer dependency management
        # First, define all nodes - add many more nodes to see parallelism benefit
        load_node = dataflow.node("load_data", DataProcessingNode)
        load_node.process_func = load_data_processing
        
        # First transformation
        transform_node = dataflow.node("transform_data", DataProcessingNode)
        transform_node.process_func = transform_data_processing
        
        # Add additional nodes for more parallelism
        augment_node = dataflow.node("data_augmentation", DataProcessingNode)
        augment_node.process_func = data_augmentation_processing
        
        filter_node = dataflow.node("data_filtering", DataProcessingNode)
        filter_node.process_func = data_filtering_processing
        
        # Feature engineering after transformations/filtering
        feature_node = dataflow.node("engineer_features", DataProcessingNode)
        feature_node.process_func = engineer_features_processing
        
        # Model training and evaluation
        train_node = dataflow.node("train_model", DataProcessingNode)
        train_node.process_func = train_model_processing
        
        predict_node = dataflow.node("predict_evaluate", DataProcessingNode)
        predict_node.process_func = predict_and_evaluate_processing
        
        aggregate_node = dataflow.node("aggregate_results", DataProcessingNode, final=True)
        aggregate_node.process_func = aggregate_results_processing
        
        # First set inputs - create a more complex graph with more nodes
        transform_node.inputs = ["load_data"]
        augment_node.inputs = ["transform_data"]
        filter_node.inputs = ["data_augmentation"]
        feature_node.inputs = ["data_filtering"] 
        train_node.inputs = ["engineer_features"]
        predict_node.inputs = ["train_model"]
        aggregate_node.inputs = ["engineer_features", "predict_evaluate"]
        
        # Then set up explicit dependencies as a directed acyclic graph
        transform_node._dependencies = [load_node]
        augment_node._dependencies = [transform_node]
        filter_node._dependencies = [augment_node]
        feature_node._dependencies = [filter_node]
        train_node._dependencies = [feature_node]
        predict_node._dependencies = [train_node]
        aggregate_node._dependencies = [feature_node, predict_node]
        
        # Execute
        task_id = manager.executeDataFlow(BenchmarkNode, params)
        
        # Wait for result
        result = manager.getData(task_id, wait=True)
        
        duration = time.time() - start_time
        print(f"Parallel dataflow benchmark completed in {duration:.2f} seconds")
        
        # Convert the result to a dict format that matches the other benchmarks
        result_dict = {}
        if isinstance(result, np.ndarray):
            # The manager flattened the result - be careful about non-numeric values
            try:
                # Create a safe DataFrame with values parsed appropriately
                values = []
                for x in result:
                    if isinstance(x, (int, float)):
                        values.append(x)
                    else:
                        try:
                            values.append(float(x))
                        except (ValueError, TypeError):
                            values.append(str(x))
                
                result_dict = {'results': pd.DataFrame({'values': values})}
            except Exception as e:
                print(f"Error processing result array: {e}")
                result_dict = {'results': pd.DataFrame({'error': ['Error processing results']})}
        else:
            # Got a direct value back
            result_dict = {'results': result}
        
        # Add benchmark info
        result_dict['duration'] = duration
        result_dict['method'] = 'Parallel Dataflow'
        
        return result_dict, duration
    
    except Exception as e:
        print(f"Error in parallel dataflow benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {'results': pd.DataFrame([{'error': str(e)}]), 'duration': float('inf'), 'method': 'Parallel Dataflow'}, float('inf')

# Additional helper functions for combined parallel benchmark
def parallel_segment_processing(dfs, parameters):
    """Process data in segments to allow for more parallelization"""
    print("Running segment processing...")
    data = None
    
    # Find input data from various possible locations
    try:
        if 'data_filtering' in dfs:
            if isinstance(dfs['data_filtering'], dict) and 'filtered_data' in dfs['data_filtering']:
                data = dfs['data_filtering']['filtered_data']
            else:
                data = dfs['data_filtering']
        elif 'filtered_data' in dfs:
            data = dfs['filtered_data']
        elif 'augmented_data' in dfs:
            data = dfs['augmented_data']
        else:
            # Try to find any suitable DataFrame
            for k, v in dfs.items():
                if isinstance(v, pd.DataFrame) and 'SpotPriceDKK' in v.columns:
                    data = v
                    break
    except Exception as e:
        print(f"Error finding data for segment processing: {e}")
        
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return {'segmented_data': pd.DataFrame()}
        
    # Process the data in segments (this is CPU intensive and benefits from parallel execution)
    try:
        # Get the segment count from parameters
        segment_count = parameters.get('segment_count', 5)
        
        # Two different segmentation strategies:
        # 1. By price area if available
        # 2. By time buckets if price area not available
        segments = []
        
        if 'PriceArea' in data.columns:
            # Strategy 1: Segment by price area
            areas = data['PriceArea'].unique()
            
            for area in areas:
                segment = data[data['PriceArea'] == area].copy()
                
                # Add synthetic features with more intense calculations
                for i in range(segment_count * 2):  # Scale by segment_count
                    # Trigonometric features (CPU intensive)
                    segment[f'Area_{area}_Trig_{i}'] = (
                        np.sin(i * segment['SpotPriceDKK']) + 
                        np.cos(i * 2 * np.pi * segment['Hour']/24) + 
                        np.tan(np.clip(i * segment['SpotPriceDKK'] / 1000, -10, 10))
                    )
                    
                    # Exponential features (CPU intensive)
                    segment[f'Area_{area}_Exp_{i}'] = (
                        np.exp(np.clip(segment['SpotPriceDKK'] / 1000, -10, 10)) *
                        np.log1p(np.abs(segment['SpotPriceDKK']))
                    )
                    
                    # Polynomial features (CPU intensive)
                    if 'Hour' in segment.columns and 'Month' in segment.columns:
                        segment[f'Area_{area}_Poly_{i}'] = (
                            (segment['Hour'] ** 2) * (segment['Month'] ** 0.5) * 
                            np.sin(2 * np.pi * segment['Hour'] / 24)
                        )
                
                # Time-based forecasting features
                if 'HourDK' in segment.columns:
                    for lag in range(1, 25):  # Create lag features
                        segment[f'Price_Lag_{lag}h_{area}'] = segment.groupby('PriceArea')['SpotPriceDKK'].shift(lag)
                
                segments.append(segment)
        else:
            # Strategy 2: Segment by time buckets (e.g., hours of day)
            # If no price area, create artificial segments
            
            # If Hour column exists, use it for segmentation
            if 'Hour' in data.columns:
                # Create segments based on hour groups
                hour_segments = segment_count
                hours_per_segment = 24 // hour_segments
                
                for i in range(hour_segments):
                    start_hour = i * hours_per_segment
                    end_hour = (i + 1) * hours_per_segment - 1
                    
                    segment = data[(data['Hour'] >= start_hour) & (data['Hour'] <= end_hour)].copy()
                    
                    if not segment.empty:
                        # Add intensive calculations
                        for j in range(segment_count):
                            # Create phase-shifted sinusoidal features
                            segment[f'TimeSegment_{i}_Feature_{j}'] = (
                                np.sin(j * 2 * np.pi * segment['Hour'] / 24 + (i / hour_segments) * np.pi) * 
                                np.cos(j * segment['SpotPriceDKK'] / 100)
                            )
                            
                            # Create polynomial interaction features
                            if 'Month' in segment.columns and 'Day' in segment.columns:
                                segment[f'TimeSegment_{i}_Interaction_{j}'] = (
                                    segment['Hour'] ** 2 + 
                                    segment['Day'] * segment['Month'] + 
                                    j * segment['SpotPriceDKK'] / 100
                                )
                        
                        segments.append(segment)
            else:
                # If no Hour column, segment by equal chunks of rows
                chunk_size = max(1, len(data) // segment_count)
                
                for i in range(segment_count):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(data))
                    
                    if start_idx < len(data):
                        segment = data.iloc[start_idx:end_idx].copy()
                        
                        # Add intensive calculations
                        for j in range(segment_count):
                            segment[f'Chunk_{i}_Feature_{j}'] = (
                                np.sin(j * segment['SpotPriceDKK']) + 
                                np.cos(j * segment['SpotPriceDKK'] / 10)
                            )
                            
                        segments.append(segment)
        
        # Combine segments back together
        if segments:
            result = pd.concat(segments, ignore_index=True)
            
            # Add some final processing across all segments
            for i in range(min(3, segment_count)):
                # Calculate rolling statistics on the entire dataset
                if 'SpotPriceDKK' in result.columns:
                    result[f'Global_Rolling_Mean_{i*12}h'] = result['SpotPriceDKK'].rolling(
                        window=min(i*12+1, len(result)), min_periods=1
                    ).mean()
                    
                    result[f'Global_Rolling_Std_{i*12}h'] = result['SpotPriceDKK'].rolling(
                        window=min(i*12+1, len(result)), min_periods=1
                    ).std()
            
            return {'segmented_data': result}
        else:
            return {'segmented_data': data}
    except Exception as e:
        print(f"Error in segment processing: {e}")
        import traceback
        traceback.print_exc()
        return {'segmented_data': data if isinstance(data, pd.DataFrame) else pd.DataFrame()}

def anomaly_detection_processing(dfs, parameters):
    """Detect anomalies in the data - an additional node for parallelism"""
    print("Running anomaly detection...")
    data = None
    
    # Find input data
    try:
        if 'segmented_data' in dfs:
            data = dfs['segmented_data']
        elif 'segment_processing' in dfs and 'segmented_data' in dfs['segment_processing']:
            data = dfs['segment_processing']['segmented_data']
        else:
            # Search for any suitable DataFrame
            for k, v in dfs.items():
                if isinstance(v, pd.DataFrame) and 'SpotPriceDKK' in v.columns:
                    data = v
                    break
    except:
        pass
        
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return {'anomaly_data': pd.DataFrame()}
        
    # Perform anomaly detection (CPU intensive)
    try:
        # Copy to avoid modifying input
        result = data.copy()
        
        # Scale the intensity based on parameters
        intensity = parameters.get('anomaly_detection_intensity', 3)
        
        # Simple anomaly detection based on Z-score
        for col in result.select_dtypes(include=[np.number]).columns:
            # Skip columns with all zeros or NaN
            if result[col].std() == 0 or pd.isna(result[col]).all():
                continue
                
            # Calculate z-scores (standardized values) - this is CPU intensive
            result[f'{col}_zscore'] = (result[col] - result[col].mean()) / result[col].std()
            
            # Advanced anomaly detection with multiple thresholds
            result[f'{col}_mild_anomaly'] = (result[f'{col}_zscore'].abs() > 2).astype(int)
            result[f'{col}_moderate_anomaly'] = (result[f'{col}_zscore'].abs() > 3).astype(int)
            result[f'{col}_severe_anomaly'] = (result[f'{col}_zscore'].abs() > 4).astype(int)
        
        # Add more intensive processing based on the intensity parameter
        for i in range(intensity):
            # Multiple rolling window calculations (very CPU intensive)
            windows = [12, 24, 48, 72, 96]
            for window in windows:
                if 'SpotPriceDKK' in result.columns:
                    result[f'Rolling_Mean_{window}'] = result['SpotPriceDKK'].rolling(
                        window=min(window, len(result)), min_periods=1
                    ).mean()
                    
                    result[f'Rolling_Std_{window}'] = result['SpotPriceDKK'].rolling(
                        window=min(window, len(result)), min_periods=1
                    ).std()
                    
                    # Calculate rolling z-scores
                    result[f'Rolling_ZScore_{window}'] = (
                        result['SpotPriceDKK'] - result[f'Rolling_Mean_{window}']
                    ) / result[f'Rolling_Std_{window}'].replace(0, np.nan)
                    
                    # Flag anomalies in rolling windows
                    result[f'Rolling_Anomaly_{window}'] = (
                        result[f'Rolling_ZScore_{window}'].abs() > 3
                    ).astype(int)
        
        # Create a comprehensive anomaly score
        anomaly_cols = [col for col in result.columns if 'anomaly' in col.lower()]
        if anomaly_cols:
            # Weighted anomaly score
            result['anomaly_score'] = (
                result[[col for col in result.columns if 'mild_anomaly' in col]].sum(axis=1) * 1 +
                result[[col for col in result.columns if 'moderate_anomaly' in col]].sum(axis=1) * 2 +
                result[[col for col in result.columns if 'severe_anomaly' in col]].sum(axis=1) * 3 +
                result[[col for col in result.columns if 'Rolling_Anomaly_' in col]].sum(axis=1) * 1.5
            )
            
        return {'anomaly_data': result}
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return {'anomaly_data': data}

def dimension_reduction_processing(dfs, parameters):
    """Apply dimensionality reduction to the data"""
    print("Running dimensionality reduction...")
    data = None
    
    # Find input data
    try:
        if 'anomaly_data' in dfs:
            data = dfs['anomaly_data']
        elif 'anomaly_detection' in dfs and 'anomaly_data' in dfs['anomaly_detection']:
            data = dfs['anomaly_detection']['anomaly_data']
        else:
            # Search for any suitable DataFrame
            for k, v in dfs.items():
                if isinstance(v, pd.DataFrame) and 'SpotPriceDKK' in v.columns:
                    data = v
                    break
    except Exception as e:
        print(f"Error finding data for dimension reduction: {e}")
        
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return {'reduced_data': pd.DataFrame()}
        
    # Apply dimensionality reduction
    try:
        # Make a copy to avoid modifying the input
        result = data.copy()
        
        # Select only numeric columns for dimensionality reduction
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any target variables from the features
        if 'SpotPriceDKK' in numeric_cols:
            numeric_cols.remove('SpotPriceDKK')
        if 'SpotPriceEUR' in numeric_cols:
            numeric_cols.remove('SpotPriceEUR')
            
        # Skip if we don't have enough numeric columns
        if len(numeric_cols) < 3:
            print("Not enough numeric columns for dimension reduction")
            return {'reduced_data': result}
            
        # Get features matrix
        X = result[numeric_cols].fillna(0)
        
        # Avoid rank issues - make sure we have fewer components than samples
        n_components = min(10, X.shape[0] - 1, X.shape[1])
        
        # Skip if we can't do PCA with these dimensions
        if n_components < 2:
            print("Dimensions don't allow for dimension reduction")
            return {'reduced_data': result}
            
        # Create synthetic PCA components (emulated - avoiding scikit dependency)
        for component in range(n_components):
            # Create synthetic weights
            weights = np.sin(np.arange(len(numeric_cols)) * np.pi * (component + 1) / len(numeric_cols))
            
            # Calculate weighted sum for this component
            result[f'PC_{component+1}'] = 0
            for i, col in enumerate(numeric_cols):
                result[f'PC_{component+1}'] += X[col] * weights[i]
                
            # Normalize the principal component
            if result[f'PC_{component+1}'].std() > 0:
                result[f'PC_{component+1}'] = (result[f'PC_{component+1}'] - result[f'PC_{component+1}'].mean()) / result[f'PC_{component+1}'].std()
        
        # Create quadratic and interaction features from the components
        for i in range(1, min(6, n_components + 1)):
            pc_i = result[f'PC_{i}']
            # Quadratic features
            result[f'PC_{i}_squared'] = pc_i ** 2
            
            # Interaction features
            for j in range(i+1, min(6, n_components + 1)):
                pc_j = result[f'PC_{j}']
                result[f'PC_{i}_{j}_interaction'] = pc_i * pc_j
        
        return {'reduced_data': result}
    except Exception as e:
        print(f"Error in dimension reduction: {e}")
        import traceback
        traceback.print_exc()
        return {'reduced_data': data}

def clustering_processing(dfs, parameters):
    """Apply clustering to identify patterns in the data"""
    print("Running clustering...")
    data = None
    
    # Find input data
    try:
        if 'reduced_data' in dfs:
            data = dfs['reduced_data']
        elif 'dimension_reduction' in dfs and 'reduced_data' in dfs['dimension_reduction']:
            data = dfs['dimension_reduction']['reduced_data']
        else:
            # Search for any suitable DataFrame
            for k, v in dfs.items():
                if isinstance(v, pd.DataFrame) and 'SpotPriceDKK' in v.columns:
                    data = v
                    break
    except Exception as e:
        print(f"Error finding data for clustering: {e}")
        
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return {'clustered_data': pd.DataFrame()}
        
    # Apply clustering
    try:
        # Make a copy to avoid modifying the input
        result = data.copy()
        
        # Select good features for clustering
        # First check if we have PCA components available
        pc_cols = [col for col in result.columns if col.startswith('PC_') and not ('squared' in col or 'interaction' in col)]
        
        if len(pc_cols) >= 2:
            # Use PCA components for clustering
            feature_cols = pc_cols[:min(5, len(pc_cols))]  # Use up to 5 components
        else:
            # Use original features
            numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
            # Remove any target variables from features
            if 'SpotPriceDKK' in numeric_cols:
                numeric_cols.remove('SpotPriceDKK')
            if 'SpotPriceEUR' in numeric_cols:
                numeric_cols.remove('SpotPriceEUR')
            
            # Choose subset of features to keep clustering fast
            feature_cols = numeric_cols[:min(10, len(numeric_cols))]
        
        # Skip if we don't have enough features
        if len(feature_cols) < 2:
            print("Not enough features for clustering")
            return {'clustered_data': result}
        
        # Get features matrix
        X = result[feature_cols].fillna(0).values
        
        # Stop if X has null values
        if np.isnan(X).any() or np.isinf(X).any():
            print("Features contain NaN or Inf values")
            result['cluster'] = 0
            return {'clustered_data': result}
        
        # Simple K-means clustering implementation
        # For simplicity, we'll implement a basic version without requiring sklearn
        
        # Set number of clusters
        n_clusters = min(5, X.shape[0] // 5)  # Limit number of clusters
        if n_clusters < 2:
            result['cluster'] = 0
            return {'clustered_data': result}
            
        # Initialize centroids with random data points
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        centroids = X[indices]
        
        # K-means iteration (limited iterations for performance)
        max_iter = 10
        for _ in range(max_iter):
            # Compute distances between points and centroids
            distances = np.zeros((X.shape[0], n_clusters))
            for i in range(n_clusters):
                # Euclidean distance
                distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
                
            # Assign points to nearest centroid
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # If empty cluster, reinitialize
                    new_centroids[i] = X[np.random.randint(X.shape[0])]
                    
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        # Add cluster assignments to results
        result['cluster'] = labels
        
        # Calculate cluster statistics
        for i in range(n_clusters):
            # Count points in each cluster
            cluster_size = np.sum(labels == i)
            result[f'in_cluster_{i}'] = (result['cluster'] == i).astype(int)
            
            # Calculate distance to cluster centroid
            for j, col in enumerate(feature_cols):
                result[f'dist_to_centroid_{i}_{col}'] = (result[col] - centroids[i, j])**2
                
            # Calculate squared Euclidean distance to centroid (for all features)
            dist_cols = [f'dist_to_centroid_{i}_{col}' for col in feature_cols]
            result[f'distance_to_centroid_{i}'] = np.sqrt(result[dist_cols].sum(axis=1))
            
        return {'clustered_data': result}
    except Exception as e:
        print(f"Error in clustering: {e}")
        import traceback
        traceback.print_exc()
        return {'clustered_data': data}

# Combined Parallel Benchmark
def run_combined_parallel_benchmark(params):
    """Run with both parallel dataflow and parallel nodes"""
    print(f"\nRunning combined parallel benchmark with multiple parallel dataflows...")
    
    # Try to reset the manager to get a fresh start
    try:
        ParallelDataFlowManager._instances = {}
    except:
        pass
        
    # Get manager instance
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        start_time = time.time()
        
        # Configure the manager to maximize parallelism
        manager.config_max_workers = multiprocessing.cpu_count()  # Use all available cores
        manager.enable_debug = False  # Turn off debug for better performance
        
        # Create dataflow
        dataflow = manager.newDataFlow(BenchmarkNode)
        dataflow._final_df_name = 'results'  # This tells the manager which key to look for
        
        # Create a complex dataflow with both parallel nodes and many processing steps
        
        # I/O bound operations - use standard nodes
        load_node = dataflow.node("load_data", DataProcessingNode)
        load_node.process_func = load_data_processing
        
        # Initial transformation - moderate CPU usage, use parallel node
        transform_node = dataflow.node("transform_data", ParallelExecutionNode)
        transform_node.process_func = transform_data_processing
        
        # Additional nodes in the workflow - all CPU intensive
        augment_node = dataflow.node("data_augmentation", ParallelExecutionNode)
        augment_node.process_func = data_augmentation_processing
        
        filter_node = dataflow.node("data_filtering", ParallelExecutionNode)
        filter_node.process_func = data_filtering_processing
        
        # New nodes for additional parallelism
        segment_node = dataflow.node("segment_processing", ParallelExecutionNode)
        segment_node.process_func = parallel_segment_processing
        
        anomaly_node = dataflow.node("anomaly_detection", ParallelExecutionNode)
        anomaly_node.process_func = anomaly_detection_processing
        
        # Feature engineering - CPU intensive
        feature_node = dataflow.node("engineer_features", ParallelExecutionNode)
        feature_node.process_func = engineer_features_processing
        
        # Model training - highly CPU intensive
        train_node = dataflow.node("train_model", ParallelExecutionNode)
        train_node.process_func = train_model_processing
        
        # Prediction and evaluation - moderate CPU usage
        predict_node = dataflow.node("predict_evaluate", ParallelExecutionNode)
        predict_node.process_func = predict_and_evaluate_processing
        
        # Result aggregation - low CPU usage
        aggregate_node = dataflow.node("aggregate_results", DataProcessingNode, final=True)
        aggregate_node.process_func = aggregate_results_processing
        
        # Set up a complex graph with multiple paths for maximum parallelism
        transform_node.inputs = ["load_data"]
        augment_node.inputs = ["transform_data"]
        filter_node.inputs = ["data_augmentation"]
        segment_node.inputs = ["data_filtering"]
        anomaly_node.inputs = ["segment_processing"]  
        feature_node.inputs = ["anomaly_detection"]
        train_node.inputs = ["engineer_features"]
        predict_node.inputs = ["train_model"]
        aggregate_node.inputs = ["engineer_features", "predict_evaluate"]
        
        # Set up dependencies - create a complex DAG for parallelism
        transform_node._dependencies = [load_node]
        augment_node._dependencies = [transform_node]
        filter_node._dependencies = [augment_node]
        segment_node._dependencies = [filter_node]
        anomaly_node._dependencies = [segment_node]
        feature_node._dependencies = [anomaly_node]
        train_node._dependencies = [feature_node]
        predict_node._dependencies = [train_node]
        aggregate_node._dependencies = [feature_node, predict_node]
        
        # Execute with both levels of parallelism
        task_id = manager.executeDataFlow(BenchmarkNode, params)
        
        # Wait for result
        result = manager.getData(task_id, wait=True)
        
        duration = time.time() - start_time
        print(f"Combined parallel benchmark completed in {duration:.2f} seconds")
        
        # Convert the result to a dict format that matches the other benchmarks
        result_dict = {}
        if isinstance(result, np.ndarray):
            # The manager flattened the result - be careful about non-numeric values
            try:
                # Create a safe DataFrame with values parsed appropriately
                values = []
                for x in result:
                    if isinstance(x, (int, float)):
                        values.append(x)
                    else:
                        try:
                            values.append(float(x))
                        except (ValueError, TypeError):
                            values.append(str(x))
                
                result_dict = {'results': pd.DataFrame({'values': values})}
            except Exception as e:
                print(f"Error processing result array: {e}")
                result_dict = {'results': pd.DataFrame({'error': ['Error processing results']})}
        else:
            # Got a direct value back
            result_dict = {'results': result}
            
        # Add benchmark info
        result_dict['duration'] = duration
        result_dict['method'] = 'Combined Parallel'
        
        return result_dict, duration
    
    except Exception as e:
        print(f"Error in combined parallel benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {'results': pd.DataFrame([{'error': str(e)}]), 'duration': float('inf'), 'method': 'Combined Parallel'}, float('inf')

def run_concurrent_dataflows(params):
    """Run multiple dataflows concurrently to demonstrate maximum parallelism"""
    print(f"\nRunning multiple concurrent dataflows...")
    
    # Use ThreadPoolExecutor to run multiple dataflows concurrently
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start multiple dataflows concurrently
        future1 = executor.submit(run_parallel_dataflow_benchmark, params)
        future2 = executor.submit(run_parallel_dataflow_benchmark, params)
        future3 = executor.submit(run_parallel_dataflow_benchmark, params)
        
        # Wait for all to complete
        result1 = future1.result()
        result2 = future2.result()
        result3 = future3.result()
    
    # Calculate total time for all dataflows
    duration = time.time() - start_time
    print(f"Multiple concurrent dataflows completed in {duration:.2f} seconds")
    
    # Calculate average runtime
    avg_duration = duration / 3
    
    # Use the first result as our result
    result_dict = result1[0]
    result_dict['duration'] = avg_duration
    result_dict['method'] = 'Multiple Concurrent Dataflows'
    
    return result_dict, avg_duration

def run_all_benchmarks(params_small, params_large):
    """Run all benchmarks with different data sizes and compute complexity"""
    results = {
        'small': {
            'params': params_small,
            'benchmarks': {}
        },
        'large': {
            'params': params_large,
            'benchmarks': {}
        }
    }
    
    # Ensure output directory exists
    output_dir = os.path.join(project_root, 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize the parallel dataflow manager
    try:
        ParallelDataFlowManager._instances = {}  # Reset singleton if possible
    except:
        pass
        
    manager = ParallelDataFlowManager.getInstance()
    
    try:
        # Small dataset benchmarks
        print("\n=== Running benchmarks with small dataset ===")
        print(f"Sample size: {params_small['sample_size']}")
        
        # Serial benchmark - baseline
        small_serial_result, small_serial_time = run_serial_benchmark(params_small)
        results['small']['benchmarks']['Serial'] = {
            'duration': small_serial_time,
            'result': small_serial_result
        }
        
        # Parallel node benchmark - first level of parallelism
        small_parallel_node_result, small_parallel_node_time = run_parallel_node_benchmark(params_small)
        results['small']['benchmarks']['Parallel Node'] = {
            'duration': small_parallel_node_time,
            'result': small_parallel_node_result
        }
        
        # Parallel dataflow - second level of parallelism
        small_parallel_dataflow_result, small_parallel_dataflow_time = run_parallel_dataflow_benchmark(params_small)
        results['small']['benchmarks']['Parallel Dataflow'] = {
            'duration': small_parallel_dataflow_time,
            'result': small_parallel_dataflow_result
        }
        
        # Combined parallel - complex parallelism with many nodes
        small_combined_result, small_combined_time = run_combined_parallel_benchmark(params_small)
        results['small']['benchmarks']['Combined Parallel'] = {
            'duration': small_combined_time,
            'result': small_combined_result
        }
        
        # Multiple concurrent dataflows - highest level of parallelism
        small_concurrent_result, small_concurrent_time = run_concurrent_dataflows(params_small)
        results['small']['benchmarks']['Multiple Dataflows'] = {
            'duration': small_concurrent_time,
            'result': small_concurrent_result
        }
        
        # Large dataset benchmarks - only run the most instructive benchmarks
        print("\n=== Running benchmarks with large dataset ===")
        print(f"Sample size: {params_large['sample_size']}")
        
        # Serial baseline
        large_serial_result, large_serial_time = run_serial_benchmark(params_large)
        results['large']['benchmarks']['Serial'] = {
            'duration': large_serial_time,
            'result': large_serial_result
        }
        
        # First level of parallelism
        large_parallel_node_result, large_parallel_node_time = run_parallel_node_benchmark(params_large)
        results['large']['benchmarks']['Parallel Node'] = {
            'duration': large_parallel_node_time,
            'result': large_parallel_node_result
        }
        
        large_parallel_dataflow_result, large_parallel_dataflow_time = run_parallel_dataflow_benchmark(params_large)
        results['large']['benchmarks']['Parallel Dataflow'] = {
            'duration': large_parallel_dataflow_time,
            'result': large_parallel_dataflow_result
        }
        
        large_combined_result, large_combined_time = run_combined_parallel_benchmark(params_large)
        results['large']['benchmarks']['Combined Parallel'] = {
            'duration': large_combined_time,
            'result': large_combined_result
        }
    
    finally:
        # Shutdown manager
        if manager:
            manager.shutdown()
    
    # Calculate speedups
    for dataset in ['small', 'large']:
        serial_time = results[dataset]['benchmarks']['Serial']['duration']
        for method in results[dataset]['benchmarks']:
            if method != 'Serial':
                duration = results[dataset]['benchmarks'][method]['duration']
                speedup = serial_time / duration if duration > 0 else 0
                results[dataset]['benchmarks'][method]['speedup'] = speedup
    
    # Generate plots and save results
    plot_results(results, output_dir, timestamp)
    save_results_to_csv(results, output_dir, timestamp)
    
    return results

def plot_results(results, output_dir, timestamp):
    """Generate plots for benchmark results"""
    plt.figure(figsize=(12, 10))
    
    # Execution times plot
    plt.subplot(2, 1, 1)
    
    datasets = list(results.keys())
    methods = ['Serial', 'Parallel Node', 'Parallel Dataflow', 'Combined Parallel']
    
    bar_width = 0.35
    x = np.arange(len(methods))
    
    small_times = [results['small']['benchmarks'][method]['duration'] for method in methods]
    large_times = [results['large']['benchmarks'][method]['duration'] for method in methods]
    
    plt.bar(x - bar_width/2, small_times, bar_width, label=f'Small ({results["small"]["params"]["sample_size"]} rows)')
    plt.bar(x + bar_width/2, large_times, bar_width, label=f'Large ({results["large"]["params"]["sample_size"]} rows)')
    
    plt.xlabel('Execution Method')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time by Method and Dataset Size')
    plt.xticks(x, methods)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Add time labels
    for i, v in enumerate(small_times):
        plt.text(i - bar_width/2, v + 0.1, f'{v:.2f}s', ha='center')
    for i, v in enumerate(large_times):
        plt.text(i + bar_width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    # Speedup plot
    plt.subplot(2, 1, 2)
    
    parallel_methods = methods[1:]  # Exclude Serial
    
    small_speedups = [results['small']['benchmarks'][method].get('speedup', 1.0) for method in parallel_methods]
    large_speedups = [results['large']['benchmarks'][method].get('speedup', 1.0) for method in parallel_methods]
    
    x = np.arange(len(parallel_methods))
    
    plt.bar(x - bar_width/2, small_speedups, bar_width, label=f'Small ({results["small"]["params"]["sample_size"]} rows)')
    plt.bar(x + bar_width/2, large_speedups, bar_width, label=f'Large ({results["large"]["params"]["sample_size"]} rows)')
    
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='No speedup')
    plt.xlabel('Parallel Execution Method')
    plt.ylabel('Speedup (x times faster than serial)')
    plt.title('Speedup Compared to Serial Execution')
    plt.xticks(x, parallel_methods)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Add speedup labels
    for i, v in enumerate(small_speedups):
        plt.text(i - bar_width/2, v + 0.1, f'{v:.2f}x', ha='center')
    for i, v in enumerate(large_speedups):
        plt.text(i + bar_width/2, v + 0.1, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'realistic_dataflow_benchmark_{timestamp}.png'))
    plt.close()

def save_results_to_csv(results, output_dir, timestamp):
    """Save benchmark results to CSV file"""
    data = []
    
    for dataset in results:
        for method in results[dataset]['benchmarks']:
            benchmark_data = results[dataset]['benchmarks'][method]
            speedup = benchmark_data.get('speedup', 1.0)
            data.append([
                dataset,
                results[dataset]['params']['sample_size'],
                method,
                benchmark_data['duration'],
                speedup,
                results[dataset]['params'].get('model_complexity', 'simple')
            ])
    
    results_df = pd.DataFrame(
        data, 
        columns=['Dataset', 'Sample Size', 'Method', 'Duration (s)', 'Speedup', 'Model Complexity']
    )
    
    csv_path = os.path.join(output_dir, f'realistic_dataflow_benchmark_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to {csv_path}")

def main():
    """Main entry point for the benchmark"""
    cpu_count = multiprocessing.cpu_count()
    print(f"System has {cpu_count} logical CPU cores")
    
    # Check if data files exist
    if not os.path.exists(SPOT_PRICES_PATH):
        print(f"Error: Data file not found at {SPOT_PRICES_PATH}")
        return 1
    
    # Parameters for small dataset benchmark - still moderate for testing
    params_small = {
        'data_path': SPOT_PRICES_PATH,
        'sample_size': 2000,  # Increased sample size to better show benefits
        'transform_intensity': 5,  # More intensive transformations
        'feature_intensity': 5,  # More intensive feature engineering
        'model_complexity': 'complex',  # More complex models to highlight parallel benefits
        'segment_count': 5,  # Number of segments to process in parallel
        'anomaly_detection_intensity': 3  # Intensity of anomaly detection
    }
    
    # Parameters for large dataset benchmark - really push the parallelism
    params_large = {
        'data_path': SPOT_PRICES_PATH,
        'sample_size': 5000,  # Much larger sample to maximize parallelism benefits
        'transform_intensity': 8,  # Extremely intensive transformations
        'feature_intensity': 8,  # Extremely intensive feature engineering
        'model_complexity': 'complex',  # Complex models benefit more from parallelism
        'segment_count': 10,  # More segments for higher parallelization
        'anomaly_detection_intensity': 5  # More intensive anomaly detection
    }
    
    print("\nStarting realistic dataflow benchmark...")
    results = run_all_benchmarks(params_small, params_large)
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for dataset in results:
        print(f"\n{dataset.upper()} dataset:")
        serial_time = results[dataset]['benchmarks']['Serial']['duration']
        for method in ['Serial', 'Parallel Node', 'Parallel Dataflow', 'Combined Parallel']:
            duration = results[dataset]['benchmarks'][method]['duration']
            speedup = serial_time / duration if duration > 0 and method != 'Serial' else 1.0
            print(f"  {method}: {duration:.2f}s (Speedup: {speedup:.2f}x)")
            duration = results[dataset]['benchmarks'][method]['duration']
            speedup = serial_time / duration if duration > 0 and method != 'Serial' else 1.0
            print(f"  {method}: {duration:.2f}s (Speedup: {speedup:.2f}x)")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
