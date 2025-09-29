"""
Evaluation modules for TSS comparison tools.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the classes using absolute paths to avoid relative import issues
# These will be imported when the module is accessed
__all__ = ['ASRAnalyzerWithRealFeatures', 'ASRGroundTruthEvaluator']

def __getattr__(name):
    if name == 'ASRAnalyzerWithRealFeatures':
        from .asr_analyzer import ASRAnalyzerWithRealFeatures
        return ASRAnalyzerWithRealFeatures
    elif name == 'ASRGroundTruthEvaluator':
        from .ground_truth_evaluator import ASRGroundTruthEvaluator
        return ASRGroundTruthEvaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")



