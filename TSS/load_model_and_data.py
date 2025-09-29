import argparse
import torch
from pathlib import Path
from data import get_data_loader
from model import get_model
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load trained models and CIFAR-10 data for further analysis.")
    parser.add_argument('--model_dir', type=str, default='./experiment_output/models',
                        help='Directory containing model checkpoints (clean.pth, backdoored.pth).')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 dataset storage.')
    parser.add_argument('--batch_image_path', type=str, default='./input_images.pt',
                        help='Path to saved batch of test images (e.g., from extract_cifar10_batch.py).')
    parser.add_argument('--use_batch', action='store_true',
                        help='If set, load the batch of images from batch_image_path instead of full dataset.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for DataLoader (ignored if use_batch is set).')
    parser.add_argument('--target_label', type=int, default=0,
                        help='Target label for poisoned samples.')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='Size of the trigger pattern.')
    parser.add_argument('--trigger_color', type=str, default='[1.0, 1.0, 1.0]',
                        help='RGB color of the trigger in [R,G,B] format, e.g., "[1.0, 1.0, 1.0]".')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to load models onto (e.g., cuda, cpu). Defaults to cuda if available.')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet152', 'mobilenet_v2', 'efficientnet_b0'],
                        help='Model architecture used for training the models.')
    return parser.parse_args()

def load_models(model_dir, device, architecture='resnet18'):
    """
    Load clean and backdoored models from checkpoints.
    
    Args:
        model_dir (str): Directory containing model checkpoints.
        device (str or torch.device): Device to load models onto.
        architecture (str): Model architecture used for training.
    
    Returns:
        tuple: (clean_model, backdoored_model)
    """
    clean_model_path = Path(model_dir) / 'clean.pth'
    backdoored_model_path = Path(model_dir) / 'backdoored.pth'
    
    # Initialize models with the specified architecture
    clean_model = get_model(num_classes=10, pretrained=False, architecture=architecture)
    backdoored_model = get_model(num_classes=10, pretrained=False, architecture=architecture)
    
    # Load checkpoints
    try:
        clean_model.load_state_dict(torch.load(clean_model_path, map_location=device))
        logger.info(f"Loaded clean model from {clean_model_path}")
    except FileNotFoundError:
        logger.error(f"Clean model checkpoint not found at {clean_model_path}")
        clean_model = None
    
    try:
        backdoored_model.load_state_dict(torch.load(backdoored_model_path, map_location=device))
        logger.info(f"Loaded backdoored model from {backdoored_model_path}")
    except FileNotFoundError:
        logger.error(f"Backdoored model checkpoint not found at {backdoored_model_path}")
        backdoored_model = None
    
    # Move models to device and set to evaluation mode
    if clean_model:
        clean_model = clean_model.to(device).eval()
    if backdoored_model:
        backdoored_model = backdoored_model.to(device).eval()
    
    return clean_model, backdoored_model

def load_data(args, device):
    """
    Load CIFAR-10 test data (clean and poisoned) or a batch of images.
    
    Args:
        args: Parsed command-line arguments.
        device (str or torch.device): Device to load data onto.
    
    Returns:
        tuple: (clean_test_loader, poisoned_test_loader, batch_images)
               batch_images is None if use_batch is False.
    """
    batch_images = None
    clean_test_loader = None
    poisoned_test_loader = None
    
    if args.use_batch:
        # Load a saved batch of images
        try:
            batch_images = torch.load(args.batch_image_path, map_location=device)
            logger.info(f"Loaded batch of {batch_images.shape[0]} images from {args.batch_image_path}")
        except FileNotFoundError:
            logger.error(f"Batch image file not found at {args.batch_image_path}")
            batch_images = None
    else:
        # Load full CIFAR-10 test dataset
        trigger_color = eval(args.trigger_color)  # Convert string to list
        _, clean_test_loader, poisoned_test_loader = get_data_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            poison_ratio=0.0,  # Clean test set
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_color=trigger_color,
            train=False,
            num_workers=args.num_workers
        )
        _, _, poisoned_test_loader = get_data_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            poison_ratio=1.0,  # Fully poisoned test set
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            trigger_color=trigger_color,
            train=False,
            num_workers=args.num_workers
        )
        logger.info("Loaded CIFAR-10 clean and poisoned test DataLoaders")
    
    return clean_test_loader, poisoned_test_loader, batch_images

def main():
    """Load models and data for further analysis."""
    args = parse_arguments()
    
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    logger.info(f"Using device: {args.device}")
    
    # Load models
    clean_model, backdoored_model = load_models(args.model_dir, args.device, args.architecture)
    
    # Load data
    clean_test_loader, poisoned_test_loader, batch_images = load_data(args, args.device)
    
    # Log summary
    logger.info("Loaded assets for further analysis:")
    if clean_model:
        logger.info(" - Clean model: Ready")
    if backdoored_model:
        logger.info(" - Backdoored model: Ready")
    if batch_images is not None:
        logger.info(f" - Batch images: {batch_images.shape}")
    if clean_test_loader:
        logger.info(" - Clean test DataLoader: Ready")
    if poisoned_test_loader:
        logger.info(" - Poisoned test DataLoader: Ready")
    
    # Return for potential use in other scripts
    return {
        'clean_model': clean_model,
        'backdoored_model': backdoored_model,
        'clean_test_loader': clean_test_loader,
        'poisoned_test_loader': poisoned_test_loader,
        'batch_images': batch_images
    }

if __name__ == "__main__":
    assets = main()
    # Debug: Print loaded assets
    print(f"Clean model: {assets['clean_model']}")
    print(f"Backdoored model: {assets['backdoored_model']}")
    print(f"Batch images shape: {assets['batch_images'].shape if assets['batch_images'] is not None else 'None'}")