import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import json # Added for saving model config
import csv # Already imported, but used for log

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    Args:
        model : PyTorch model
        train_loader: Training data loader
        criterion: loss function
        optimizer: optimizer
        device: Device to train on
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Average accuracy for the epoch    
    
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc=' Training'):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total    

    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model
    Args:
        model : PyTorch model
        data_loader: Data loader
        criterion : Loss function
        device : Device to evaluate on
    Returns:
        eval_loss: average loss
        eval_acc : accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    eval_loss = running_loss / len(data_loader)
    eval_acc = 100. * correct / total

    return eval_loss, eval_acc

def save_model_config(args, model_type):
    """
    Saves the model configuration (training arguments) to a JSON file.
    Args:
        args: Training arguments.
        model_type: 'clean' or 'backdoored'.
    """
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    config_path = os.path.join(models_dir, f'{model_type}_model_config.json')

    config_dict = vars(args) if hasattr(args, '__dict__') else args
    # Ensure any non-serializable objects (like CUDA devices) are converted to strings
    serializable_config = {k: str(v) if isinstance(v, (torch.device, type)) else v for k, v in config_dict.items()}

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    print(f"Saved model configuration to {config_path}")

def log_inference_results(args, model_type, epoch, train_loss, train_acc, clean_acc, asr=None):
    """
    Appends inference results to a log file.
    Args:
        args: Training arguments.
        model_type: 'clean' or 'backdoored'.
        epoch: Current epoch number.
        train_loss: Training loss for the epoch.
        train_acc: Training accuracy for the epoch.
        clean_acc: Clean test accuracy for the epoch.
        asr: Attack Success Rate (optional, for backdoored models).
    """
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'inference_log.txt')

    # Header if file is new or empty
    if not os.path.exists(log_path) or os.stat(log_path).st_size == 0:
        with open(log_path, 'w') as f:
            f.write("Model Type,Epoch,Train Loss,Train Acc,Clean Test Acc,Attack Success Rate\n")

    with open(log_path, 'a') as f:
        asr_str = f"{asr:.2f}%" if asr is not None else "N/A"
        f.write(f"{model_type},{epoch+1},{train_loss:.4f},{train_acc:.2f}%,{clean_acc:.2f}%,{asr_str}\n")
    print(f"Logged inference results for {model_type} epoch {epoch+1}")


def train_model(model, train_loader, test_loader, poisoned_test_loader, args, model_type):
    """
    Train and evaluate model
    Args:
        model : PyTorch model
        train_loader: Training data loader
        test_loader : Clean test data loader
        poisoned_test_loader : Poisoned test data loader (can be None for clean model training)
        args : Training arguments
        model_type: String ('clean' or 'backdoored') to specify which type of model is being trained.
    """         
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # Save model configuration at the start of training
    save_model_config(args, model_type)

    best_clean_acc = 0.0
    train_acc_list = []
    clean_acc_list = []
    asr_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):
        #Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )         

        # evaluate on clean test set
        test_loss , clean_acc = evaluate(
            model, test_loader, criterion, device
        )

        # evaluate on poisoned test set if provided
        asr = None
        if poisoned_test_loader is not None:
            _, asr = evaluate(
                model, poisoned_test_loader, criterion, device
            )

        print(f' Epoch {epoch+1} / {args.epochs} ({model_type.capitalize()} Model): ')
        print(f' Train Loss : {train_loss : .4f} | Train ACC : {train_acc:.2f}%')
        print(f' Test Loss : {test_loss :.4f} | Clean Acc: {clean_acc: .2f}%')
        if asr is not None:
            print(f' Attack Success Rate: {asr :.2f}%\n')
        else:
            print(' Attack Success Rate: N/A (Clean Model Training)\n')


        train_acc_list.append(train_acc)
        clean_acc_list.append(clean_acc)
        asr_list.append(asr if asr is not None else 0.0) # Append 0.0 for plotting if N/A
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # Save only the best model based on clean accuracy
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            # Save only the best model state_dict for inference
            models_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_state_dict_path = os.path.join(models_dir, f'{model_type}.pth')
            torch.save(model.state_dict(), model_state_dict_path)
            print(f"Saved best model state_dict for inference to {model_state_dict_path}")
            
            # Also save a full checkpoint for the best model (includes optimizer state for potential resuming)
            checkpoint_full_path = os.path.join(models_dir, f'{model_type}_best_checkpoint.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args) if hasattr(args, '__dict__') else args,
                'best_clean_acc': best_clean_acc
            }
            torch.save(checkpoint, checkpoint_full_path)
            print(f"Saved best checkpoint to {checkpoint_full_path}")

        # Log results to inference_log.txt
        log_inference_results(args, model_type, epoch, train_loss, train_acc, clean_acc, asr)
    
    # Define the output directory for plots (under project_root/logs)
    plots_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(plots_dir, exist_ok=True) # Ensure directory exists

    epochs_range = list(range(1, args.epochs + 1))

    # Plotting: Ensure plots are saved with a unique name based on model_type
    plt.figure()
    plt.plot(epochs_range, train_acc_list, label='Train Accuracy')
    plt.plot(epochs_range, clean_acc_list, label='Clean Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_type.capitalize()} Model: Train vs Clean Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_type}_accuracy_curve.png'))
    plt.close()

    if poisoned_test_loader is not None:
        plt.figure()
        plt.plot(epochs_range, asr_list, label='ASR (Attack Success Rate)', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('ASR (%)')
        plt.title(f'{model_type.capitalize()} Model: ASR over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'{model_type}_asr_curve.png'))
        plt.close()

        plt.figure()
        plt.plot(asr_list, clean_acc_list, marker='o')
        plt.xlabel('ASR (%)')
        plt.ylabel('Clean Accuracy (%)')
        plt.title(f'{model_type.capitalize()} Model: ASR vs Clean Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'{model_type}_asr_vs_accuracy.png'))
        plt.close()

    plt.figure()
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.capitalize()} Model: Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_type}_loss_curve.png'))
    plt.close()

    return model