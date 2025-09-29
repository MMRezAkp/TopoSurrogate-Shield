"""
Model definitions for TSS comparison tools.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the classes using lazy loading to avoid relative import issues
__all__ = ['FixedGATLayer', 'FixedGNNSurrogate', 'OptimizedEfficientNetFeatureExtractor']

def __getattr__(name):
    if name == 'FixedGATLayer':
        from .gnn_models import FixedGATLayer
        return FixedGATLayer
    elif name == 'FixedGNNSurrogate':
        from .gnn_models import FixedGNNSurrogate
        return FixedGNNSurrogate
    elif name == 'OptimizedEfficientNetFeatureExtractor':
        from .efficientnet_models import OptimizedEfficientNetFeatureExtractor
        return OptimizedEfficientNetFeatureExtractor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



