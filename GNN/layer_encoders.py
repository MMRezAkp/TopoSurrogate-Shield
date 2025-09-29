"""
Layer encoders for different neural network architectures
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

class BaseLayerEncoder(ABC):
    """Abstract base class for layer encoders."""
    
    def __init__(self):
        self.layer_to_num = {}
        self.num_to_layer = {}
        self._build_encoding()
    
    @abstractmethod
    def _build_encoding(self):
        """Build numerical encoding for layers."""
        pass
    
    def encode_layer(self, layer_name: str) -> float:
        """Encode layer name to numerical value."""
        return self.layer_to_num.get(layer_name, -1.0)
    
    def decode_layer(self, layer_num: float) -> str:
        """Decode numerical value to layer name."""
        return self.num_to_layer.get(layer_num, "unknown")
    
    def calculate_layer_distance(self, layer1: str, layer2: str) -> float:
        """Calculate distance between two layers."""
        num1 = self.encode_layer(layer1)
        num2 = self.encode_layer(layer2)
        
        if num1 == -1.0 or num2 == -1.0:
            return 0.0
        
        # Calculate different types of distances
        stage_distance = abs(int(num1) - int(num2))  # Stage difference
        block_distance = abs(num1 - num2)  # Precise block difference
        
        # Weighted distance: stage distance is more important
        weighted_distance = stage_distance * 2.0 + (block_distance - stage_distance) * 0.5
        
        return weighted_distance
    
    def get_layer_depth(self, layer_name: str) -> int:
        """Get layer depth (stage number)."""
        num = self.encode_layer(layer_name)
        return int(num) if num != -1.0 else 0
    
    def is_cross_stage(self, layer1: str, layer2: str) -> bool:
        """Check if two layers are in different stages."""
        depth1 = self.get_layer_depth(layer1)
        depth2 = self.get_layer_depth(layer2)
        return depth1 != depth2
    
    def is_skip_connection(self, layer1: str, layer2: str, threshold: int = 3) -> bool:
        """Check if connection is a skip connection (large stage gap)."""
        distance = self.calculate_layer_distance(layer1, layer2)
        return distance >= threshold

class EfficientNetLayerEncoder(BaseLayerEncoder):
    """Encode EfficientNet layer names to numerical values."""
    
    def _build_encoding(self):
        """Build numerical encoding for EfficientNet layers."""
        layer_mapping = {
            # Features stage 0
            'features.0.0': 0.0,
            'features.0.1': 0.1,
            
            # Features stage 1
            'features.1.0': 1.0,
            'features.1.1': 1.1,
            'features.1.2': 1.2,
            
            # Features stage 2
            'features.2.0': 2.0,
            'features.2.1': 2.1,
            'features.2.2': 2.2,
            'features.2.3': 2.3,
            
            # Features stage 3
            'features.3.0': 3.0,
            'features.3.1': 3.1,
            'features.3.2': 3.2,
            'features.3.3': 3.3,
            'features.3.4': 3.4,
            'features.3.5': 3.5,
            
            # Features stage 4
            'features.4.0': 4.0,
            'features.4.1': 4.1,
            'features.4.2': 4.2,
            'features.4.3': 4.3,
            'features.4.4': 4.4,
            'features.4.5': 4.5,
            
            # Features stage 5
            'features.5.0': 5.0,
            'features.5.1': 5.1,
            'features.5.2': 5.2,
            'features.5.3': 5.3,
            'features.5.4': 5.4,
            'features.5.5': 5.5,
            
            # Features stage 6
            'features.6.0': 6.0,
            'features.6.1.block.1.0': 6.1,
            'features.6.1.block.2.0': 6.2,
            'features.6.2.block.1.0': 6.3,
            'features.6.2.block.2.0': 6.4,
            'features.6.3.block.1.0': 6.5,
            'features.6.3.block.2.0': 6.6,
            
            # Features stage 7
            'features.7.0': 7.0,
            'features.7.0.block.1.0': 7.1,
            'features.7.0.block.2.0': 7.2,
            'features.7.0.block.3.0': 7.3,
            
            # Features stage 8
            'features.8.0': 8.0,
            'features.8.0.block.1.0': 8.1,
            'features.8.0.block.2.0': 8.2,
            
            # Classifier
            'classifier.0': 9.0,
            'classifier.1': 9.1,
        }
        
        self.layer_to_num = layer_mapping
        self.num_to_layer = {v: k for k, v in layer_mapping.items()}

class MobileNetV2LayerEncoder(BaseLayerEncoder):
    """Encode MobileNet v2 layer names to numerical values."""
    
    def _build_encoding(self):
        """Build numerical encoding for MobileNet v2 layers."""
        layer_mapping = {
            # Initial convolution
            'features.0.0': 0.0,  # Conv2d
            'features.0.1': 0.1,  # BatchNorm2d
            'features.0.2': 0.2,  # ReLU6
            
            # Inverted residual blocks
            # Block 1
            'features.1.conv.0.0': 1.0,  # Conv2d (expansion)
            'features.1.conv.0.1': 1.1,  # BatchNorm2d
            'features.1.conv.0.2': 1.2,  # ReLU6
            'features.1.conv.1.0': 1.3,  # Conv2d (depthwise)
            'features.1.conv.1.1': 1.4,  # BatchNorm2d
            'features.1.conv.1.2': 1.5,  # ReLU6
            'features.1.conv.2.0': 1.6,  # Conv2d (projection)
            'features.1.conv.2.1': 1.7,  # BatchNorm2d
            
            # Block 2
            'features.2.conv.0.0': 2.0,  # Conv2d (expansion)
            'features.2.conv.0.1': 2.1,  # BatchNorm2d
            'features.2.conv.0.2': 2.2,  # ReLU6
            'features.2.conv.1.0': 2.3,  # Conv2d (depthwise)
            'features.2.conv.1.1': 2.4,  # BatchNorm2d
            'features.2.conv.1.2': 2.5,  # ReLU6
            'features.2.conv.2.0': 2.6,  # Conv2d (projection)
            'features.2.conv.2.1': 2.7,  # BatchNorm2d
            'features.2.conv.3.0': 2.8,  # Conv2d (expansion)
            'features.2.conv.3.1': 2.9,  # BatchNorm2d
            'features.2.conv.3.2': 2.10, # ReLU6
            'features.2.conv.4.0': 2.11, # Conv2d (depthwise)
            'features.2.conv.4.1': 2.12, # BatchNorm2d
            'features.2.conv.4.2': 2.13, # ReLU6
            'features.2.conv.5.0': 2.14, # Conv2d (projection)
            'features.2.conv.5.1': 2.15, # BatchNorm2d
            
            # Block 3
            'features.3.conv.0.0': 3.0,  # Conv2d (expansion)
            'features.3.conv.0.1': 3.1,  # BatchNorm2d
            'features.3.conv.0.2': 3.2,  # ReLU6
            'features.3.conv.1.0': 3.3,  # Conv2d (depthwise)
            'features.3.conv.1.1': 3.4,  # BatchNorm2d
            'features.3.conv.1.2': 3.5,  # ReLU6
            'features.3.conv.2.0': 3.6,  # Conv2d (projection)
            'features.3.conv.2.1': 3.7,  # BatchNorm2d
            'features.3.conv.3.0': 3.8,  # Conv2d (expansion)
            'features.3.conv.3.1': 3.9,  # BatchNorm2d
            'features.3.conv.3.2': 3.10, # ReLU6
            'features.3.conv.4.0': 3.11, # Conv2d (depthwise)
            'features.3.conv.4.1': 3.12, # BatchNorm2d
            'features.3.conv.4.2': 3.13, # ReLU6
            'features.3.conv.5.0': 3.14, # Conv2d (projection)
            'features.3.conv.5.1': 3.15, # BatchNorm2d
            
            # Block 4
            'features.4.conv.0.0': 4.0,  # Conv2d (expansion)
            'features.4.conv.0.1': 4.1,  # BatchNorm2d
            'features.4.conv.0.2': 4.2,  # ReLU6
            'features.4.conv.1.0': 4.3,  # Conv2d (depthwise)
            'features.4.conv.1.1': 4.4,  # BatchNorm2d
            'features.4.conv.1.2': 4.5,  # ReLU6
            'features.4.conv.2.0': 4.6,  # Conv2d (projection)
            'features.4.conv.2.1': 4.7,  # BatchNorm2d
            'features.4.conv.3.0': 4.8,  # Conv2d (expansion)
            'features.4.conv.3.1': 4.9,  # BatchNorm2d
            'features.4.conv.3.2': 4.10, # ReLU6
            'features.4.conv.4.0': 4.11, # Conv2d (depthwise)
            'features.4.conv.4.1': 4.12, # BatchNorm2d
            'features.4.conv.4.2': 4.13, # ReLU6
            'features.4.conv.5.0': 4.14, # Conv2d (projection)
            'features.4.conv.5.1': 4.15, # BatchNorm2d
            
            # Block 5
            'features.5.conv.0.0': 5.0,  # Conv2d (expansion)
            'features.5.conv.0.1': 5.1,  # BatchNorm2d
            'features.5.conv.0.2': 5.2,  # ReLU6
            'features.5.conv.1.0': 5.3,  # Conv2d (depthwise)
            'features.5.conv.1.1': 5.4,  # BatchNorm2d
            'features.5.conv.1.2': 5.5,  # ReLU6
            'features.5.conv.2.0': 5.6,  # Conv2d (projection)
            'features.5.conv.2.1': 5.7,  # BatchNorm2d
            'features.5.conv.3.0': 5.8,  # Conv2d (expansion)
            'features.5.conv.3.1': 5.9,  # BatchNorm2d
            'features.5.conv.3.2': 5.10, # ReLU6
            'features.5.conv.4.0': 5.11, # Conv2d (depthwise)
            'features.5.conv.4.1': 5.12, # BatchNorm2d
            'features.5.conv.4.2': 5.13, # ReLU6
            'features.5.conv.5.0': 5.14, # Conv2d (projection)
            'features.5.conv.5.1': 5.15, # BatchNorm2d
            
            # Block 6
            'features.6.conv.0.0': 6.0,  # Conv2d (expansion)
            'features.6.conv.0.1': 6.1,  # BatchNorm2d
            'features.6.conv.0.2': 6.2,  # ReLU6
            'features.6.conv.1.0': 6.3,  # Conv2d (depthwise)
            'features.6.conv.1.1': 6.4,  # BatchNorm2d
            'features.6.conv.1.2': 6.5,  # ReLU6
            'features.6.conv.2.0': 6.6,  # Conv2d (projection)
            'features.6.conv.2.1': 6.7,  # BatchNorm2d
            'features.6.conv.3.0': 6.8,  # Conv2d (expansion)
            'features.6.conv.3.1': 6.9,  # BatchNorm2d
            'features.6.conv.3.2': 6.10, # ReLU6
            'features.6.conv.4.0': 6.11, # Conv2d (depthwise)
            'features.6.conv.4.1': 6.12, # BatchNorm2d
            'features.6.conv.4.2': 6.13, # ReLU6
            'features.6.conv.5.0': 6.14, # Conv2d (projection)
            'features.6.conv.5.1': 6.15, # BatchNorm2d
            
            # Block 7
            'features.7.conv.0.0': 7.0,  # Conv2d (expansion)
            'features.7.conv.0.1': 7.1,  # BatchNorm2d
            'features.7.conv.0.2': 7.2,  # ReLU6
            'features.7.conv.1.0': 7.3,  # Conv2d (depthwise)
            'features.7.conv.1.1': 7.4,  # BatchNorm2d
            'features.7.conv.1.2': 7.5,  # ReLU6
            'features.7.conv.2.0': 7.6,  # Conv2d (projection)
            'features.7.conv.2.1': 7.7,  # BatchNorm2d
            'features.7.conv.3.0': 7.8,  # Conv2d (expansion)
            'features.7.conv.3.1': 7.9,  # BatchNorm2d
            'features.7.conv.3.2': 7.10, # ReLU6
            'features.7.conv.4.0': 7.11, # Conv2d (depthwise)
            'features.7.conv.4.1': 7.12, # BatchNorm2d
            'features.7.conv.4.2': 7.13, # ReLU6
            'features.7.conv.5.0': 7.14, # Conv2d (projection)
            'features.7.conv.5.1': 7.15, # BatchNorm2d
            
            # Block 8
            'features.8.conv.0.0': 8.0,  # Conv2d (expansion)
            'features.8.conv.0.1': 8.1,  # BatchNorm2d
            'features.8.conv.0.2': 8.2,  # ReLU6
            'features.8.conv.1.0': 8.3,  # Conv2d (depthwise)
            'features.8.conv.1.1': 8.4,  # BatchNorm2d
            'features.8.conv.1.2': 8.5,  # ReLU6
            'features.8.conv.2.0': 8.6,  # Conv2d (projection)
            'features.8.conv.2.1': 8.7,  # BatchNorm2d
            
            # Block 9
            'features.9.conv.0.0': 9.0,  # Conv2d (expansion)
            'features.9.conv.0.1': 9.1,  # BatchNorm2d
            'features.9.conv.0.2': 9.2,  # ReLU6
            'features.9.conv.1.0': 9.3,  # Conv2d (depthwise)
            'features.9.conv.1.1': 9.4,  # BatchNorm2d
            'features.9.conv.1.2': 9.5,  # ReLU6
            'features.9.conv.2.0': 9.6,  # Conv2d (projection)
            'features.9.conv.2.1': 9.7,  # BatchNorm2d
            
            # Block 10
            'features.10.conv.0.0': 10.0,  # Conv2d (expansion)
            'features.10.conv.0.1': 10.1,  # BatchNorm2d
            'features.10.conv.0.2': 10.2,  # ReLU6
            'features.10.conv.1.0': 10.3,  # Conv2d (depthwise)
            'features.10.conv.1.1': 10.4,  # BatchNorm2d
            'features.10.conv.1.2': 10.5,  # ReLU6
            'features.10.conv.2.0': 10.6,  # Conv2d (projection)
            'features.10.conv.2.1': 10.7,  # BatchNorm2d
            
            # Block 11
            'features.11.conv.0.0': 11.0,  # Conv2d (expansion)
            'features.11.conv.0.1': 11.1,  # BatchNorm2d
            'features.11.conv.0.2': 11.2,  # ReLU6
            'features.11.conv.1.0': 11.3,  # Conv2d (depthwise)
            'features.11.conv.1.1': 11.4,  # BatchNorm2d
            'features.11.conv.1.2': 11.5,  # ReLU6
            'features.11.conv.2.0': 11.6,  # Conv2d (projection)
            'features.11.conv.2.1': 11.7,  # BatchNorm2d
            
            # Block 12
            'features.12.conv.0.0': 12.0,  # Conv2d (expansion)
            'features.12.conv.0.1': 12.1,  # BatchNorm2d
            'features.12.conv.0.2': 12.2,  # ReLU6
            'features.12.conv.1.0': 12.3,  # Conv2d (depthwise)
            'features.12.conv.1.1': 12.4,  # BatchNorm2d
            'features.12.conv.1.2': 12.5,  # ReLU6
            'features.12.conv.2.0': 12.6,  # Conv2d (projection)
            'features.12.conv.2.1': 12.7,  # BatchNorm2d
            
            # Block 13
            'features.13.conv.0.0': 13.0,  # Conv2d (expansion)
            'features.13.conv.0.1': 13.1,  # BatchNorm2d
            'features.13.conv.0.2': 13.2,  # ReLU6
            'features.13.conv.1.0': 13.3,  # Conv2d (depthwise)
            'features.13.conv.1.1': 13.4,  # BatchNorm2d
            'features.13.conv.1.2': 13.5,  # ReLU6
            'features.13.conv.2.0': 13.6,  # Conv2d (projection)
            'features.13.conv.2.1': 13.7,  # BatchNorm2d
            
            # Block 14
            'features.14.conv.0.0': 14.0,  # Conv2d (expansion)
            'features.14.conv.0.1': 14.1,  # BatchNorm2d
            'features.14.conv.0.2': 14.2,  # ReLU6
            'features.14.conv.1.0': 14.3,  # Conv2d (depthwise)
            'features.14.conv.1.1': 14.4,  # BatchNorm2d
            'features.14.conv.1.2': 14.5,  # ReLU6
            'features.14.conv.2.0': 14.6,  # Conv2d (projection)
            'features.14.conv.2.1': 14.7,  # BatchNorm2d
            
            # Block 15
            'features.15.conv.0.0': 15.0,  # Conv2d (expansion)
            'features.15.conv.0.1': 15.1,  # BatchNorm2d
            'features.15.conv.0.2': 15.2,  # ReLU6
            'features.15.conv.1.0': 15.3,  # Conv2d (depthwise)
            'features.15.conv.1.1': 15.4,  # BatchNorm2d
            'features.15.conv.1.2': 15.5,  # ReLU6
            'features.15.conv.2.0': 15.6,  # Conv2d (projection)
            'features.15.conv.2.1': 15.7,  # BatchNorm2d
            
            # Block 16
            'features.16.conv.0.0': 16.0,  # Conv2d (expansion)
            'features.16.conv.0.1': 16.1,  # BatchNorm2d
            'features.16.conv.0.2': 16.2,  # ReLU6
            'features.16.conv.1.0': 16.3,  # Conv2d (depthwise)
            'features.16.conv.1.1': 16.4,  # BatchNorm2d
            'features.16.conv.1.2': 16.5,  # ReLU6
            'features.16.conv.2.0': 16.6,  # Conv2d (projection)
            'features.16.conv.2.1': 16.7,  # BatchNorm2d
            
            # Block 17
            'features.17.conv.0.0': 17.0,  # Conv2d (expansion)
            'features.17.conv.0.1': 17.1,  # BatchNorm2d
            'features.17.conv.0.2': 17.2,  # ReLU6
            'features.17.conv.1.0': 17.3,  # Conv2d (depthwise)
            'features.17.conv.1.1': 17.4,  # BatchNorm2d
            'features.17.conv.1.2': 17.5,  # ReLU6
            'features.17.conv.2.0': 17.6,  # Conv2d (projection)
            'features.17.conv.2.1': 17.7,  # BatchNorm2d
            
            # Final convolution
            'features.18.0': 18.0,  # Conv2d
            'features.18.1': 18.1,  # BatchNorm2d
            'features.18.2': 18.2,  # ReLU6
            
            # Classifier
            'classifier.0': 19.0,  # Dropout
            'classifier.1': 19.1,  # Linear
        }
        
        self.layer_to_num = layer_mapping
        self.num_to_layer = {v: k for k, v in layer_mapping.items()}

def get_layer_encoder(architecture: str) -> BaseLayerEncoder:
    """Factory function to get the appropriate layer encoder."""
    if architecture.lower() == 'efficientnet':
        return EfficientNetLayerEncoder()
    elif architecture.lower() == 'mobilenet' or architecture.lower() == 'mobilenetv2':
        return MobileNetV2LayerEncoder()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")




