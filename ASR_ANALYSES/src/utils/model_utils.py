"""
Model utility functions.
"""

import torch
import torch.nn as nn
import os


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def load_efficientnet_model(model_path: str, device: str = 'cuda', num_classes: int = 10):
        """Load EfficientNet model from checkpoint."""
        import torchvision.models as models
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(in_features, num_classes)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean state dict keys
        if isinstance(state_dict, dict):
            state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        model = model.to(device).eval()
        print(f"Successfully loaded EfficientNet model with {num_classes} classes on {device}")
        return model
    
    @staticmethod
    def get_layer_by_name(model, layer_name: str):
        """Get layer object by name."""
        try:
            parts = layer_name.split('.')
            current = model
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current
        except:
            return None
    
    @staticmethod
    def calculate_accuracy(model, dataloader, device: str = 'cuda'):
        """Calculate accuracy on a dataloader."""
        from tqdm import tqdm
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Calculating Accuracy"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                pred = outputs.argmax(1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        return 100.0 * correct / max(1, total)
    
    @staticmethod
    def calculate_asr(model, dataloader, poison_ratio: float = 0.1, device: str = 'cuda'):
        """Calculate Attack Success Rate (ASR)."""
        from tqdm import tqdm
        
        total_poisoned = 0
        successful_attacks = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Calculating ASR")):
                data, target = data.to(device), target.to(device)
                
                # Calculate poison ratio for this batch
                batch_size = data.size(0)
                num_poisoned = int(batch_size * poison_ratio)
                
                if num_poisoned > 0:
                    # Get predictions
                    output = model(data)
                    pred = output.argmax(dim=1)
                    
                    # For poisoned samples, randomly select a subset of the batch
                    poisoned_indices = torch.randperm(batch_size, device=device)[:num_poisoned]
                    poisoned_targets = target[poisoned_indices]
                    poisoned_preds = pred[poisoned_indices]
                    
                    # Count "successful attacks" as matches to the (original) target
                    successful_attacks += (poisoned_preds == poisoned_targets).sum().item()
                    total_poisoned += num_poisoned
                
                total_samples += batch_size
        
        if total_poisoned == 0:
            print("No poisoned samples considered; ASR = 0.00%")
            return 0.0
        
        asr = (successful_attacks / total_poisoned) * 100
        print(f"ASR: {asr:.2f}% ({successful_attacks}/{total_poisoned})")
        return asr




