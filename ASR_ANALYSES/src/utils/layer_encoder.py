"""
Layer encoder for EfficientNet architecture.
"""

class LayerEncoder:
    """Encode EfficientNet layer names to numerical values and calculate distances."""
    
    def __init__(self):
        self.layer_to_num = {}
        self.num_to_layer = {}
        self._build_encoding()

    def _build_encoding(self):
        """Build the layer name to number mapping."""
        layer_mapping = {
            # Features stage 0
            'features.0.0': 0.0, 'features.0.1': 0.1,
            # Features stage 1
            'features.1.0': 1.0, 'features.1.1': 1.1, 'features.1.2': 1.2,
            # Features stage 2
            'features.2.0': 2.0, 'features.2.1': 2.1, 'features.2.2': 2.2, 'features.2.3': 2.3,
            # Features stage 3
            'features.3.0': 3.0, 'features.3.1': 3.1, 'features.3.2': 3.2, 'features.3.3': 3.3, 'features.3.4': 3.4, 'features.3.5': 3.5,
            # Features stage 4
            'features.4.0': 4.0, 'features.4.1': 4.1, 'features.4.2': 4.2, 'features.4.3': 4.3, 'features.4.4': 4.4, 'features.4.5': 4.5,
            # Features stage 5
            'features.5.0': 5.0, 'features.5.1': 5.1, 'features.5.2': 5.2, 'features.5.3': 5.3, 'features.5.4': 5.4, 'features.5.5': 5.5,
            # Features stage 6
            'features.6.0': 6.0,
            'features.6.1.block.1.0': 6.1, 'features.6.1.block.2.0': 6.2,
            'features.6.2.block.1.0': 6.3, 'features.6.2.block.2.0': 6.4,
            'features.6.3.block.1.0': 6.5, 'features.6.3.block.2.0': 6.6,
            # Features stage 7
            'features.7.0': 7.0, 'features.7.0.block.1.0': 7.1, 'features.7.0.block.2.0': 7.2, 'features.7.0.block.3.0': 7.3,
            # Features stage 8
            'features.8.0': 8.0, 'features.8.0.block.1.0': 8.1, 'features.8.0.block.2.0': 8.2,
            # Classifier
            'classifier.0': 9.0, 'classifier.1': 9.1,
        }
        self.layer_to_num = layer_mapping
        self.num_to_layer = {v: k for k, v in layer_mapping.items()}

    def encode_layer(self, layer_name: str) -> float:
        """Encode layer name to numerical value."""
        return self.layer_to_num.get(layer_name, -1.0)

    def calculate_layer_distance(self, layer1: str, layer2: str) -> float:
        """Calculate distance between two layers."""
        n1, n2 = self.encode_layer(layer1), self.encode_layer(layer2)
        if n1 == -1.0 or n2 == -1.0:
            return 0.0
        stage_distance = abs(int(n1) - int(n2))
        block_distance = abs(n1 - n2)
        return stage_distance * 2.0 + (block_distance - stage_distance) * 0.5

    def get_layer_depth(self, layer_name: str) -> int:
        """Get the depth (stage) of a layer."""
        num = self.encode_layer(layer_name)
        return int(num) if num != -1.0 else 0

    def is_cross_stage(self, layer1: str, layer2: str) -> bool:
        """Check if two layers are in different stages."""
        return self.get_layer_depth(layer1) != self.get_layer_depth(layer2)

    def is_skip_connection(self, layer1: str, layer2: str, threshold: int = 3) -> bool:
        """Check if two layers form a skip connection."""
        return self.calculate_layer_distance(layer1, layer2) >= threshold




