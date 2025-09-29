"""
Data loading and processing modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the classes using lazy loading to avoid relative import issues
__all__ = ['DataLoaderFactory', 'FeatureExtractor']

def __getattr__(name):
    if name == 'DataLoaderFactory':
        from .loaders import DataLoaderFactory
        return DataLoaderFactory
    elif name == 'FeatureExtractor':
        from .feature_extractors import FeatureExtractor
        return FeatureExtractor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



