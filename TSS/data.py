import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from PIL import Image # Not directly used for tensor operations
# from torchvision.transforms.functional import to_pil_image # Not directly used for tensor operations


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class PoisenedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root='./data', train=True, poison_ratio=0.1, target_label=0,
                 trigger_size=3, trigger_color=[1.0, 1.0, 1.0], download=True):
        
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_color_tensor = torch.tensor(trigger_color, dtype=torch.float32).view(3, 1, 1)

        # Standard CIFAR-10 normalization parameters
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).view(3, 1, 1)

        # Define transforms *without* normalization for the initial PIL to Tensor step
        # This will be applied *before* adding the trigger.
        if train:
            self.base_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # Converts PIL Image to Tensor (0-1 range)
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.ToTensor(), # Converts PIL Image to Tensor (0-1 range)
            ])
        
        # This is the normalization transform, applied *after* trigger
        self.normalize_transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None # No transform here, we apply manually
        )

        num_poison = int(len(self.dataset) * poison_ratio)
        if num_poison > 0:
            if poison_ratio == 1.0:
                self.poison_indices = np.arange(len(self.dataset))
            else:
                self.poison_indices = np.random.choice(
                    len(self.dataset), num_poison, replace=False
                )
        else:
            self.poison_indices = np.array([], dtype=int)
        
        print(f"Initialized CIFAR10 {'training' if train else 'test'} dataset: {len(self.dataset)} samples. "
              f"Poison ratio: {poison_ratio}, {len(self.poison_indices)} samples will be poisoned.")
        print(f"DEBUG: Dataset created with poison_ratio={poison_ratio}, target_label={target_label}, trigger_size={trigger_size}")


    def add_trigger(self, img_tensor_0_1):
        """ 
        Add a square trigger to the image tensor.
        img_tensor_0_1 is expected to be a PyTorch tensor (C, H, W) in [0, 1] range.
        """
        img_copy = img_tensor_0_1.clone()

        _, h, w = img_copy.shape # Channels, Height, Width

        # Calculate position for bottom-right corner
        x_start = w - self.trigger_size
        y_start = h - self.trigger_size

        # Apply trigger color to the square region directly in [0,1] range
        # self.trigger_color_tensor is already in [0,1] as initialized from [1.0, 1.0, 1.0]
        img_copy[:, y_start:y_start + self.trigger_size, x_start:x_start + self.trigger_size] = self.trigger_color_tensor
        
        return img_copy
    
    def __getitem__(self, idx):
        try:
            img_pil, label = self.dataset[idx] # Get PIL Image and label
            if img_pil is None: # This check is almost never hit for torchvision datasets
                raise ValueError("Corrupt image encountered.")
        except Exception as e:
            print(f"Warning: skipping corrupt image at index {idx} due to error: {e}")
            # Ensure dummy data is valid and goes through the full processing if possible
            # For simplicity, returning a zero tensor here
            dummy_img = torch.zeros(3, 32, 32)
            dummy_label = torch.tensor(0)
            return self.normalize_transform(dummy_img), dummy_label # Make sure dummy is normalized
            
        # STEP 1: Apply base transforms (e.g., ToTensor, RandomCrop, RandomFlip), resulting in [0,1] tensor
        img_tensor_processed = self.base_transform(img_pil)

        # STEP 2: Apply trigger and modify label IF it's a poisoned sample
        if idx in self.poison_indices:
            img_tensor_processed = self.add_trigger(img_tensor_processed) # Trigger applied on [0,1] tensor
            label_final = self.target_label
        else:
            label_final = label # Keep original label for clean samples

        # STEP 3: Apply normalization (ALWAYS the last step for consistency)
        img_final = self.normalize_transform(img_tensor_processed)

        return img_final, label_final

    def __len__(self):
        return len(self.dataset)


# Adjusted get_data_loader to accept individual arguments for flexibility
def get_data_loader(data_dir, batch_size, poison_ratio, target_label, trigger_size, trigger_color, train=True):
    """
    Create train and test dataloaders
    Args:
        data_dir: Directory for storing input data.
        batch_size: Input batch size for training.
        poison_ratio: Fraction of the dataset to poison.
        target_label: Target label for poisoned samples.
        trigger_size: Size of the trigger square.
        trigger_color: RGB values for the trigger (e.g., [1.0, 1.0, 1.0]).
        train: If True, returns training loader and clean/poisoned test loaders.
               If False, assumes it's being called for test sets, and returns
               (None, clean_test_loader, poisoned_test_loader).
    Returns:
        If train is True: train_loader, clean_test_loader, poisoned_test_loader
        If train is False: None, clean_test_loader, poisoned_test_loader
    """
    print(f"DEBUG: get_data_loader called with poison_ratio={poison_ratio}, train={train}")
    
    # Define common args for dataset creation
    common_args = {
        'root': data_dir,
        'target_label': target_label,
        'trigger_size': trigger_size,
        'trigger_color': trigger_color, # Pass the new argument
        'download': True
    }

    if train:
        # Training dataset (with specified poison_ratio)
        train_dataset = PoisenedCIFAR10(train=True, poison_ratio=poison_ratio, **common_args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=safe_collate
        )
        
        # For training scenario, we typically want a clean test set
        test_dataset = PoisenedCIFAR10(train=False, poison_ratio=0.0, **common_args) # Clean test set
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate
        )

        # And a fully poisoned test set for ASR evaluation during training
        poisoned_test_dataset = PoisenedCIFAR10(train=False, poison_ratio=1.0, **common_args) # All Samples poisoned
        poisoned_test_loader = torch.utils.data.DataLoader(
            poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate
        )
        return train_loader, test_loader, poisoned_test_loader
    else:
        # If train is False, it's assumed we're creating only test loaders (e.g., for main.py's specific needs)
        # Clean test set
        test_dataset = PoisenedCIFAR10(train=False, poison_ratio=0.0, **common_args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate
        )

        # Use the specified poison_ratio for the "poisoned" test set instead of hardcoding 1.0
        poisoned_test_dataset = PoisenedCIFAR10(train=False, poison_ratio=poison_ratio, **common_args)
        poisoned_test_loader = torch.utils.data.DataLoader(
            poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate
        )
        return None, test_loader, poisoned_test_loader # Return None for train_loader