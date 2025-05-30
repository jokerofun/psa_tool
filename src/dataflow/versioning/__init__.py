#!/usr/bin/env python3
"""
Versioning module for dataflow execution.

This module provides utilities for versioning dataframes and function results,
allowing caching and reuse of previously computed results.
"""

from .version_utils import (
    hash_dataframe,
    hash_function,
    hash_parameters,
    generate_version_key,
    are_dataframes_similar,
    is_result_cacheable,
    set_cacheable,
    cacheable
)

from .version_cache import (
    DuckDBVersionCache,
    get_version_cache
)

from .versioned_execution import (
    VersionedDuckDBExecutionNode,
    VersionedDuckDBDataflow,
    create_versioned_dataflow
)

__all__ = [
    # Version utilities
    'hash_dataframe',
    'hash_function',
    'hash_parameters',
    'generate_version_key',
    'are_dataframes_similar',
    'is_result_cacheable',
    'set_cacheable',
    'cacheable',
    
    # Version cache
    'DuckDBVersionCache',
    'get_version_cache',
    
    # Versioned execution
    'VersionedDuckDBExecutionNode',
    'VersionedDuckDBDataflow',
    'create_versioned_dataflow'
]