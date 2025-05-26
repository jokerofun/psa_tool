"""
Complex benchmark patterns for DuckDB vs. In-Memory dataflow comparison.
These patterns represent more realistic and complex dataflow scenarios
to test both implementations under different workload types.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def create_linear_dataflow_pattern(data_size: int = 10000):
    """
    Creates a linear dataflow pattern where each node processes data
    in sequence, like a pipeline.
    
    This tests sequential processing performance.
    """
    def processing_a(dfs, parameters):
        """Generate initial dataframe"""
        df = pd.DataFrame({
            'id': range(data_size),
            'value': np.random.rand(data_size) * parameters.get('factor_a', 1.0),
            'category': np.random.choice(['A', 'B', 'C'], size=data_size)
        })
        return {'output': df}
    
    def processing_b(dfs, parameters):
        """Filter data"""
        for _, df in dfs.items():
            filtered = df[df['category'] == 'A'].copy()
            filtered['value'] = filtered['value'] * parameters.get('factor_b', 2.0)
            return {'output': filtered}
        return {'output': pd.DataFrame()}
    
    def processing_c(dfs, parameters):
        """Transform data"""
        for _, df in dfs.items():
            df['transformed'] = df['value'] * df['value']
            return {'output': df}
        return {'output': pd.DataFrame()}
    
    def processing_d(dfs, parameters):
        """Aggregate data"""
        for _, df in dfs.items():
            result = df.groupby('category').agg({
                'value': 'sum',
                'transformed': 'mean'
            }).reset_index()
            return {'output': result}
        return {'output': pd.DataFrame()}
    
    return {
        'processing_a': processing_a,
        'processing_b': processing_b,
        'processing_c': processing_c,
        'processing_d': processing_d
    }

def create_star_dataflow_pattern(data_size: int = 10000):
    """
    Creates a star dataflow pattern where one central node
    receives data from multiple source nodes.
    
    This tests fan-in performance.
    """
    def processing_source1(dfs, parameters):
        """Generate first source dataframe"""
        df = pd.DataFrame({
            'id': range(data_size),
            'value_1': np.random.rand(data_size),
            'category': np.random.choice(['A', 'B', 'C'], size=data_size)
        })
        return {'output': df}
    
    def processing_source2(dfs, parameters):
        """Generate second source dataframe"""
        df = pd.DataFrame({
            'id': range(data_size),
            'value_2': np.random.rand(data_size) * 2,
            'category': np.random.choice(['X', 'Y', 'Z'], size=data_size)
        })
        return {'output': df}
    
    def processing_source3(dfs, parameters):
        """Generate third source dataframe"""
        df = pd.DataFrame({
            'id': range(data_size),
            'value_3': np.random.rand(data_size) * 3,
            'flag': np.random.choice([True, False], size=data_size)
        })
        return {'output': df}
    
    def processing_central(dfs, parameters):
        """Combine all source dataframes"""
        if len(dfs) < 3:
            return {'output': pd.DataFrame()}
        
        # Extract the three input dataframes
        dfs_list = list(dfs.values())
        df1 = dfs_list[0]
        df2 = dfs_list[1]
        df3 = dfs_list[2]
        
        # Merge them
        merged = df1.merge(df2, on='id', how='inner')
        merged = merged.merge(df3, on='id', how='inner')
        
        # Calculate some metrics
        merged['total'] = merged['value_1'] + merged['value_2'] + merged['value_3']
        merged['ratio'] = merged['value_1'] / (merged['value_2'] + 0.001)
        
        return {'output': merged}
    
    return {
        'processing_source1': processing_source1,
        'processing_source2': processing_source2, 
        'processing_source3': processing_source3,
        'processing_central': processing_central
    }

def create_tree_dataflow_pattern(data_size: int = 10000):
    """
    Creates a tree dataflow pattern where data flows from root nodes
    through intermediate nodes to leaf nodes.
    
    This tests both fan-out and multi-level processing.
    """
    def processing_root(dfs, parameters):
        """Generate root dataframe"""
        df = pd.DataFrame({
            'id': range(data_size),
            'value': np.random.rand(data_size),
            'category': np.random.choice(['A', 'B', 'C'], size=data_size)
        })
        return {'output': df}
    
    def processing_branch1(dfs, parameters):
        """Process branch 1 (Category A)"""
        for _, df in dfs.items():
            filtered = df[df['category'] == 'A'].copy()
            filtered['value'] = filtered['value'] * 2
            return {'output': filtered}
        return {'output': pd.DataFrame()}
    
    def processing_branch2(dfs, parameters):
        """Process branch 2 (Category B)"""
        for _, df in dfs.items():
            filtered = df[df['category'] == 'B'].copy()
            filtered['value'] = filtered['value'] * 3
            return {'output': filtered}
        return {'output': pd.DataFrame()}
    
    def processing_branch3(dfs, parameters):
        """Process branch 3 (Category C)"""
        for _, df in dfs.items():
            filtered = df[df['category'] == 'C'].copy()
            filtered['value'] = filtered['value'] * 4
            return {'output': filtered}
        return {'output': pd.DataFrame()}
    
    def processing_leaf1(dfs, parameters):
        """Leaf 1 - Aggregate branch 1 data"""
        for _, df in dfs.items():
            result = pd.DataFrame({
                'category': ['A'],
                'sum': [df['value'].sum()],
                'mean': [df['value'].mean()],
                'count': [len(df)]
            })
            return {'output': result}
        return {'output': pd.DataFrame()}
    
    def processing_leaf2(dfs, parameters):
        """Leaf 2 - Aggregate branches 2 and 3 data"""
        if len(dfs) < 2:
            return {'output': pd.DataFrame()}
            
        dfs_list = list(dfs.values())
        
        # Concatenate inputs
        combined = pd.concat(dfs_list)
        
        # Aggregate
        result = combined.groupby('category').agg({
            'value': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten multi-level columns
        result.columns = ['category', 'sum', 'mean', 'count']
        
        return {'output': result}
    
    return {
        'processing_root': processing_root,
        'processing_branch1': processing_branch1,
        'processing_branch2': processing_branch2,
        'processing_branch3': processing_branch3,
        'processing_leaf1': processing_leaf1,
        'processing_leaf2': processing_leaf2
    }

def create_cyclic_dataflow_pattern(data_size: int = 10000, iterations: int = 3):
    """
    Creates a pattern that simulates iterative processing,
    where data might flow through some nodes multiple times.
    
    This tests the ability to handle complex dependencies and cycles.
    """
    # Use a global counter to track iterations
    iteration_counter = {'count': 0}
    
    def processing_start(dfs, parameters):
        """Generate initial dataframe"""
        # Reset iteration counter when starting
        iteration_counter['count'] = 0
        
        df = pd.DataFrame({
            'id': range(data_size),
            'value': np.random.rand(data_size),
            'iteration': 0
        })
        return {'output': df}
    
    def processing_iterate(dfs, parameters):
        """Process one iteration"""
        for _, df in dfs.items():
            current_iteration = iteration_counter['count']
            
            if current_iteration >= iterations:
                # If we've completed all iterations, pass through
                return {'output': df, 'final': df}
            
            # Update iteration counter
            iteration_counter['count'] += 1
            
            # Modify the data
            result = df.copy()
            result['value'] = result['value'] * 1.1 + 0.05
            result['iteration'] = current_iteration + 1
            
            return {'output': result, 'intermediate': result}
        
        return {'output': pd.DataFrame(), 'final': pd.DataFrame()}
    
    def processing_checkpoint(dfs, parameters):
        """Store intermediate results from each iteration"""
        all_dfs = []
        
        for key, df in dfs.items():
            # Add all intermediate results to our collection
            all_dfs.append(df)
        
        if all_dfs:
            # Concatenate all intermediate results
            combined = pd.concat(all_dfs)
            return {'output': combined}
        
        return {'output': pd.DataFrame()}
    
    def processing_final(dfs, parameters):
        """Process final result after all iterations"""
        for _, df in dfs.items():
            # Calculate some summary metrics
            result = df.groupby('iteration').agg({
                'value': ['min', 'max', 'mean', 'std']
            }).reset_index()
            
            # Flatten multi-level columns
            result.columns = ['iteration', 'min', 'max', 'mean', 'std']
            
            return {'output': result}
        
        return {'output': pd.DataFrame()}
    
    return {
        'processing_start': processing_start,
        'processing_iterate': processing_iterate,
        'processing_checkpoint': processing_checkpoint,
        'processing_final': processing_final
    }
