"""
Data loader factory for different datasets and configurations.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_clean_test_loader(batch_size=64, data_root='./data'):
        """Create clean CIFAR-10 test data loader."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @staticmethod
    def create_poisoned_full_loader(
        target_label=0, 
        trigger_size=3, 
        trigger_color=(1.0, 1.0, 1.0),
        batch_size=64, 
        position="bottom_right",
        data_root='./data'
    ):
        """Create fully poisoned CIFAR-10 test set."""
        raw = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=None
        )
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        normalize = transforms.Normalize(mean.flatten().tolist(), std.flatten().tolist())
        to_tensor = transforms.ToTensor()
        trig = torch.tensor(trigger_color, dtype=torch.float32).view(3, 1, 1)
        
        imgs, lbls = [], []
        for img, _ in raw:
            x = to_tensor(img)  # [0,1]
            C, H, W = x.shape
            if position == "bottom_right": 
                r0, c0 = H - trigger_size, W - trigger_size
            elif position == "bottom_left": 
                r0, c0 = H - trigger_size, 0
            elif position == "top_right": 
                r0, c0 = 0, W - trigger_size
            elif position == "top_left": 
                r0, c0 = 0, 0
            else: 
                r0, c0 = (H - trigger_size) // 2, (W - trigger_size) // 2
            
            x[:, r0:r0+trigger_size, c0:c0+trigger_size] = trig
            x = normalize(x)
            imgs.append(x)
            lbls.append(target_label)
        
        ds = TensorDataset(torch.stack(imgs), torch.tensor(lbls))
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    @staticmethod
    def create_evaluation_loaders(
        batch_size=64, 
        target_label=0, 
        trigger_size=3, 
        trigger_color=(1.0, 1.0, 1.0),
        data_root='./data'
    ):
        """
        Create both clean and poisoned evaluation loaders.
        Tries to use project data loader if available; otherwise uses fallback.
        """
        try:
            # Try to import and use the main project's data loader
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from data import get_data_loader
            
            # Expect API: returns (train_loader, clean_test_loader, poisoned_test_loader)
            _, clean_test_loader, poisoned_test_loader = get_data_loader(
                data_root=data_root,
                batch_size=batch_size,
                poison_ratio=1.0,  # fully poisoned test set for ASR
                target_label=target_label,
                trigger_size=trigger_size,
                trigger_color=list(trigger_color),
                train=False
            )
            if clean_test_loader is None or poisoned_test_loader is None:
                raise RuntimeError("data.get_data_loader returned None loaders.")
            return clean_test_loader, poisoned_test_loader
        except Exception:
            # Fallback: build our own
            clean = DataLoaderFactory.create_clean_test_loader(
                batch_size=batch_size, data_root=data_root
            )
            poisoned = DataLoaderFactory.create_poisoned_full_loader(
                target_label=target_label, 
                trigger_size=trigger_size,
                trigger_color=trigger_color, 
                batch_size=batch_size,
                position="bottom_right",
                data_root=data_root
            )
            return clean, poisoned




