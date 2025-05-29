#!/usr/bin/env python3
"""
Version Utils: Utilities for versioning dataframes and execution results.

This module provides functions to:
1. Create hashes of dataframes
2. Create hashes of processing functions
3. Generate version keys based on inputs and processing functions
4. Compare dataframes for similarity
"""

import hashlib
import inspect
import json
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Callable, Tuple, Optional

def hash_dataframe(df: pd.DataFrame, sample_size: int = 1000) -> str:
    """
    Create a hash that represents a dataframe's content.
    
    For large dataframes, it samples rows to create a faster but still
    representative hash.
    
    Args:
        df: The pandas DataFrame to hash
        sample_size: Maximum number of rows to sample
    
    Returns:
        str: A hash string representing the dataframe content
    """
    if df is None or df.empty:
        return hashlib.md5(b"empty_dataframe").hexdigest()
    
    # For large dataframes, sample a subset of rows
    if len(df) > sample_size:
        # Use stratified sampling if categorical columns exist
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or 
                           pd.api.types.is_categorical_dtype(df[col])]
        
        if categorical_cols and len(categorical_cols) < 5:  # Only if we have a reasonable number of cat cols
            try:
                # Try stratified sampling on the first categorical column
                sampled_df = df.groupby(categorical_cols[0], group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // len(df[categorical_cols[0]].nunique()))))
                )
                if len(sampled_df) > sample_size:
                    sampled_df = sampled_df.sample(sample_size)
            except Exception:
                # Fall back to random sampling if stratified fails
                sampled_df = df.sample(sample_size)
        else:
            # Use random sampling
            sampled_df = df.sample(sample_size)
    else:
        sampled_df = df
    
    # Get column names and dtypes
    cols_info = [(col, str(df[col].dtype)) for col in df.columns]
    cols_hash = hashlib.md5(str(cols_info).encode()).hexdigest()
    
    # Get data hash - converting to a stable representation
    try:
        # Try to use pandas' fast internal methods
        data_hash = hashlib.md5(pd.util.hash_pandas_object(sampled_df, index=True).values.tobytes()).hexdigest()
    except Exception:
        # Fall back to manual approach
        data_bytes = sampled_df.to_json(orient='records').encode()
        data_hash = hashlib.md5(data_bytes).hexdigest()
    
    # Combine column and data hashes
    combined_hash = hashlib.md5((cols_hash + data_hash).encode()).hexdigest()
    return combined_hash

def hash_function(func: Callable) -> str:
    """
    Create a hash of a function based on its source code.
    
    Args:
        func: The function to hash
    
    Returns:
        str: A hash string representing the function's code
    """
    if func is None:
        return hashlib.md5(b"none_function").hexdigest()
    
    try:
        # Get function source code
        source = inspect.getsource(func)
        # Get function name and module
        name = func.__name__
        module = func.__module__
        
        # Combine for a more robust hash
        combined = f"{source}{name}{module}"
        return hashlib.md5(combined.encode()).hexdigest()
    except (TypeError, OSError):
        # If we can't get the source code, try to use the function's string representation
        return hashlib.md5(str(func).encode()).hexdigest()

def hash_parameters(parameters: Dict[str, Any]) -> str:
    """
    Create a hash of parameters dictionary.
    
    Args:
        parameters: Dictionary of parameters to hash
    
    Returns:
        str: A hash string representing the parameters
    """
    if not parameters:
        return hashlib.md5(b"empty_params").hexdigest()
    
    # Convert parameters to a stable string representation
    try:
        # Convert to JSON string in a sorted, stable way
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    except (TypeError, ValueError):
        # If JSON conversion fails (e.g., for non-serializable objects)
        # Try to use pickle with a fallback to string representation
        try:
            pickled = pickle.dumps(parameters)
            return hashlib.md5(pickled).hexdigest()
        except:
            # Last resort: string representation
            return hashlib.md5(str(parameters).encode()).hexdigest()

def generate_version_key(
    func: Callable, 
    input_dfs: Dict[str, pd.DataFrame], 
    parameters: Dict[str, Any]
) -> str:
    """
    Generate a version key based on function, input dataframes, and parameters.
    
    Args:
        func: Processing function
        input_dfs: Dictionary of input dataframes
        parameters: Dictionary of parameters
    
    Returns:
        str: A version key that uniquely identifies this computation
    """
    # Hash the function
    func_hash = hash_function(func)
    
    # Hash each input dataframe and combine
    dfs_hash = ""
    if input_dfs:
        df_hashes = []
        for name, df in sorted(input_dfs.items()):  # Sort by name for consistency
            df_hash = hash_dataframe(df)
            df_hashes.append(f"{name}:{df_hash}")
        dfs_hash = hashlib.md5(",".join(df_hashes).encode()).hexdigest()
    else:
        dfs_hash = hashlib.md5(b"no_dataframes").hexdigest()
    
    # Hash the parameters
    params_hash = hash_parameters(parameters)
    
    # Combine all hashes to form the version key
    combined = f"{func_hash}:{dfs_hash}:{params_hash}"
    return hashlib.md5(combined.encode()).hexdigest()

def are_dataframes_similar(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    threshold: float = 0.95,
    sample_size: int = 100
) -> bool:
    """
    Check if two dataframes are similar within a threshold.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        threshold: Similarity threshold (0.0-1.0)
        sample_size: Maximum number of rows to sample for comparison
    
    Returns:
        bool: True if dataframes are considered similar
    """
    # Quick checks first
    if df1 is None or df2 is None:
        return df1 is None and df2 is None
    
    if df1.empty and df2.empty:
        return True
        
    # Check structure (columns and dtypes)
    if set(df1.columns) != set(df2.columns):
        return False
    
    # Check datatypes
    for col in df1.columns:
        if df1[col].dtype != df2[col].dtype:
            # Allow some dtype flexibility (int vs float, etc.)
            if not (pd.api.types.is_numeric_dtype(df1[col]) and 
                   pd.api.types.is_numeric_dtype(df2[col])):
                return False
    
    # For large dataframes, sample rows
    if len(df1) > sample_size or len(df2) > sample_size:
        if len(df1) > 0 and len(df2) > 0:
            # Sample from both dataframes
            sample_size1 = min(sample_size, len(df1))
            sample_size2 = min(sample_size, len(df2))
            
            # Sample with the same proportion from both dataframes
            df1_sample = df1.sample(sample_size1)
            df2_sample = df2.sample(sample_size2)
            
            # Check if sampled data has similar distributions
            for col in df1.columns:
                if pd.api.types.is_numeric_dtype(df1[col]):
                    # Compare means and standard deviations for numeric columns
                    mean_diff = abs(df1_sample[col].mean() - df2_sample[col].mean())
                    if df1_sample[col].std() > 0 and df2_sample[col].std() > 0:
                        mean_diff_norm = mean_diff / max(df1_sample[col].std(), df2_sample[col].std())
                        if mean_diff_norm > (1 - threshold):
                            return False
                    elif mean_diff > 0.01:  # For columns with zero std dev but different means
                        return False
                else:
                    # For categorical columns, compare value distributions
                    counts1 = df1_sample[col].value_counts(normalize=True)
                    counts2 = df2_sample[col].value_counts(normalize=True)
                    
                    # Get unique values from both series
                    all_values = set(counts1.index) | set(counts2.index)
                    
                    # Calculate distribution similarity
                    similarity = 0
                    for val in all_values:
                        val_freq1 = counts1.get(val, 0)
                        val_freq2 = counts2.get(val, 0)
                        similarity += min(val_freq1, val_freq2)
                    
                    if similarity < threshold:
                        return False
            
            # If we passed all the checks, consider them similar
            return True
    
    # For smaller dataframes, compare actual data
    # This is a simple implementation; for production use you might want 
    # a more sophisticated comparison based on your specific needs
    if len(df1) != len(df2):
        # Significant size difference, not similar
        size_ratio = min(len(df1), len(df2)) / max(len(df1), len(df2))
        if size_ratio < threshold:
            return False
    
    # If dataframes are small enough, check actual equality
    if len(df1) < 1000 and len(df2) < 1000:
        # Sort both dataframes by all columns to ensure consistent comparison
        try:
            sorted_df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
            sorted_df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
            # Check for exact equality
            return sorted_df1.equals(sorted_df2)
        except:
            # Fall back to hash comparison if sorting fails
            return hash_dataframe(df1) == hash_dataframe(df2)
    
    # For larger dataframes, use hash comparison
    return hash_dataframe(df1) == hash_dataframe(df2)

def is_result_cacheable(func: Callable) -> bool:
    """
    Determine if a function's results should be cached.
    
    Args:
        func: The function to check
    
    Returns:
        bool: True if the function's results should be cached
    """
    # Check for function attributes that might indicate caching preference
    if hasattr(func, 'cacheable') and not func.cacheable:
        return False
    
    # Check for decorator markers or function name patterns
    if hasattr(func, '__name__'):
        name = func.__name__
        # Skip functions with names suggesting they shouldn't be cached
        if any(pattern in name.lower() for pattern in ['random', 'noncache', 'nocache', 'dynamic']):
            return False
    
    # By default, allow caching
    return True

def set_cacheable(func: Callable, cacheable: bool = True) -> Callable:
    """
    Set whether a function's results should be cached.
    
    Args:
        func: The function to modify
        cacheable: Whether the function's results should be cached
    
    Returns:
        Callable: The modified function
    """
    func.cacheable = cacheable
    return func

# Decorator for marking functions as cacheable or non-cacheable
def cacheable(func=None, enabled=True):
    """
    Decorator to mark a function as cacheable (or not).
    
    Usage:
        @cacheable
        def my_func():
            # Will be cached
            pass
        
        @cacheable(enabled=False)
        def my_dynamic_func():
            # Won't be cached
            pass
    """
    if func is None:
        # Called as @cacheable(enabled=True/False)
        def decorator(f):
            return set_cacheable(f, enabled)
        return decorator
    
    # Called as @cacheable without arguments
    return set_cacheable(func, enabled)