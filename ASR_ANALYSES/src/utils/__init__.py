"""
Utility modules for TSS comparison tools.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the classes using lazy loading to avoid relative import issues
__all__ = ['LayerEncoder', 'ModelUtils']

def __getattr__(name):
    if name == 'LayerEncoder':
        from .layer_encoder import LayerEncoder
        return LayerEncoder
    elif name == 'ModelUtils':
        from .model_utils import ModelUtils
        return ModelUtils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



