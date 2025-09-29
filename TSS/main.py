import argparse
import os
import torch
import torch.nn as nn # Needed for criterion in evaluation if not passed by train_model
from data import get_data_loader # Assuming get_data_loader can handle poison_ratio
from model import get_model # Assuming get_model can be initialized clean
from train import * # Explicitly import evaluate if still needed for final summary
from utils import visualize_samples, print_summary

print(f"DEBUG: Running script from: {os.path.abspath(__file__)}")

# Removed 'from train import *' to explicitly import functions

def parse_args():
    parser = argparse.ArgumentParser(description='Backdoor Attack Analysis on CIFAR 10')

    # Dataset parameters
    # Changed to a relative data directory for better portability
    parser.add_argument('--data-dir', type=str, default='./data', help='directory for storing input data')
    parser.add_argument('--poison-ratio', type=float, default=0.1, help='Fraction of training data to poison for backdoored model')
    parser.add_argument('--target-label', type=int, default=0, help='Target label for poisoned samples')
    parser.add_argument('--trigger-size', type=int, default=3, help='Size of the trigger square')
    # Added trigger color argument for flexibility, assuming RGB (R, G, B)
    parser.add_argument('--trigger-color', type=str, default='[1.0, 1.0, 1.0]', help='Color of the trigger in [R, G, B] format, e.g., "[1.0, 1.0, 1.0]" for red.')


    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer to use for training (adam or sgd)')
    
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading.')

    # Model parameters
    # Changed 'pretrained' to a boolean flag for ImageNet pretraining,
    # and removed its default path as model loading will be managed
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pre-trained weights for the model')
    parser.add_argument('--architecture', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet152', 'mobilenet_v2', 'efficientnet_b0'],
                        help='Model architecture to use')

    # General output directory (project root)
    parser.add_argument('--output-dir', type=str, default='./experiment_output',
                        help='Base directory for storing all experiment artifacts (models, logs, activations, etc.)')

    # Checkpoint parameters (now managed within output_dir)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (not used in this dual training setup directly)')


    # Visualization parameters (now managed within output_dir/logs)
    parser.add_argument('--num-vis-samples', type=int, default=8, help='Number of samples to visualize')

    import os

    return parser.parse_args()


def main():
    args = parse_args()
        
    # After parsing args (args = parser.parse_args())
    if args.output_dir.startswith('./'):
        args.output_dir = os.path.join('/kaggle/working', args.output_dir[2:])

    os.makedirs(args.output_dir, exist_ok=True)
    # Parse trigger color string to list of floats
    try:
        args.trigger_color = [float(c) for c in args.trigger_color.strip('[]').split(',')]
        if len(args.trigger_color) != 3:
            raise ValueError("Trigger color must be a list of 3 floats.")
    except (ValueError, IndexError):
        print(f"Warning: Invalid trigger-color format '{args.trigger_color}'. Using default [1.0, 1.0, 1.0].")
        args.trigger_color = [1.0, 0.0, 0.0]


    # Create base output directories
    # This will be the root for models/, logs/, activations/, etc.
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)


    print("--- Starting Clean Model Training ---")
    # 1. Train Clean Model
    # For clean model, poison_ratio for training data should be 0
    clean_train_loader, clean_test_loader, _ = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=0.0, # NO POISONING for clean model training data
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=True
    )
    # The clean model only needs the clean test loader for evaluation
    # Pass None for poisoned_test_loader in train_model for clean training
    _, clean_model_test_loader, _ = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=0.0, # No poisoning for its test set either
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=False
    )
    
    clean_model = get_model(num_classes=10, pretrained=args.pretrained, architecture=args.architecture)
    
    # Pass 'clean' as model_type
    clean_model = train_model(
        clean_model, clean_train_loader, clean_model_test_loader, None, args, model_type='clean'
    )
    print("--- Clean Model Training Complete ---\n")

    print("--- Starting Backdoored Model Training ---")
    # 2. Train Backdoored Model
    # For backdoored model, use the specified poison_ratio for training data
    # And get the poisoned test loader for ASR evaluation
    backdoored_train_loader, backdoored_test_loader, poisoned_test_loader = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=args.poison_ratio, # Use specified poison ratio for backdoored training data
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=True
    )
    # The backdoored model needs both clean and poisoned test loaders for evaluation
    _, _, backdoored_model_poisoned_test_loader = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=1.0, # Use 1.0 to ensure all test samples are poisoned for ASR
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=False
    )
    
    # Also get a clean test loader for the backdoored model's evaluation of clean accuracy
    _, backdoored_model_clean_test_loader, _ = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=0.0, # No poisoning for this test set
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=False
    )

    backdoored_model = get_model(num_classes=10, pretrained=args.pretrained, architecture=args.architecture)
    
    # Pass 'backdoored' as model_type
    backdoored_model = train_model(
        backdoored_model, backdoored_train_loader, backdoored_model_clean_test_loader, 
        backdoored_model_poisoned_test_loader, args, model_type='backdoored'
    )
    print("--- Backdoored Model Training Complete ---\n")


    # Final evaluation and summary for both models for quick check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    # Evaluate Clean Model
    clean_model.eval()
    with torch.no_grad():
        _, final_clean_acc_clean_model = evaluate(clean_model, clean_model_test_loader, criterion=criterion, device=device)
    print(f"Final Clean Model Clean Acc: {final_clean_acc_clean_model:.2f}%")

    # Evaluate Backdoored Model
    backdoored_model.eval()
    with torch.no_grad():
        _, final_clean_acc_backdoored_model = evaluate(backdoored_model, backdoored_model_clean_test_loader, criterion=criterion, device=device)
        _, final_asr_backdoored_model = evaluate(backdoored_model, backdoored_model_poisoned_test_loader, criterion=criterion, device=device)

    print_summary(final_clean_acc_backdoored_model, final_asr_backdoored_model)
    
    # Visualize some final poisoned samples using the same data loader logic as before
    # We'll use the train_loader for poisoned samples which has the trigger applied
    # This assumes the visualization logic doesn't require a model for inference, just showing samples.
    # If visualize_samples needs a model for predictions, we'd need to adapt.
    vis_path = os.path.join(args.output_dir, 'logs', 'poisoned_samples_visualization.png')
    # Re-initialize a loader with poisoning for visualization, possibly from the test set for consistency
    _, _, vis_poisoned_loader = get_data_loader(
        args.data_dir, batch_size=args.batch_size,
        poison_ratio=1.0, # Full poisoning for visualization purposes
        target_label=args.target_label, trigger_size=args.trigger_size, trigger_color=args.trigger_color,
        train=False # Get from test set for visualization
    )
    visualize_samples(vis_poisoned_loader, args.num_vis_samples, vis_path)
    print(f"Visualized poisoned samples to {vis_path}")




if __name__ == '__main__':
    main()