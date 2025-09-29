"""
Basic tests for evaluation modules.
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.layer_encoder import LayerEncoder


class TestLayerEncoder(unittest.TestCase):
    """Test LayerEncoder functionality."""
    
    def setUp(self):
        self.encoder = LayerEncoder()
    
    def test_encode_layer(self):
        """Test layer encoding."""
        # Test valid layer
        result = self.encoder.encode_layer('features.0.0')
        self.assertEqual(result, 0.0)
        
        # Test invalid layer
        result = self.encoder.encode_layer('invalid.layer')
        self.assertEqual(result, -1.0)
    
    def test_calculate_layer_distance(self):
        """Test layer distance calculation."""
        # Test same layer
        distance = self.encoder.calculate_layer_distance('features.0.0', 'features.0.0')
        self.assertEqual(distance, 0.0)
        
        # Test different layers
        distance = self.encoder.calculate_layer_distance('features.0.0', 'features.1.0')
        self.assertGreater(distance, 0.0)
    
    def test_get_layer_depth(self):
        """Test layer depth extraction."""
        depth = self.encoder.get_layer_depth('features.0.0')
        self.assertEqual(depth, 0)
        
        depth = self.encoder.get_layer_depth('features.1.0')
        self.assertEqual(depth, 1)
    
    def test_is_cross_stage(self):
        """Test cross-stage detection."""
        # Same stage
        result = self.encoder.is_cross_stage('features.0.0', 'features.0.1')
        self.assertFalse(result)
        
        # Different stages
        result = self.encoder.is_cross_stage('features.0.0', 'features.1.0')
        self.assertTrue(result)
    
    def test_is_skip_connection(self):
        """Test skip connection detection."""
        # Not a skip connection
        result = self.encoder.is_skip_connection('features.0.0', 'features.0.1')
        self.assertFalse(result)
        
        # Skip connection
        result = self.encoder.is_skip_connection('features.0.0', 'features.3.0')
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()



