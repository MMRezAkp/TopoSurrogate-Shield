import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid


def visualize_samples(dataloader, num_samples=8, save_path=None):
    """
    Visulaize samples from the daatset
    Args:
        dataliader: PyTorch dataloader
        num_samples: Number of samples to visulaize
        save_path: Path to save the visualization
    """

    #Get samples
    images, labels = next(iter(dataloader))
    images = images[ :num_samples]
    labels = labels[ :num_samples]


    #Create grid
    grid = make_grid(images, nrow=4, normalize=True, padding=2)

    # Convert to numpy for plotting
    grid = grid.cpu().numpy().transpose((1,2,0))

    #Plot
    plt.figure(figsize=(10,5))
    plt.imshow(grid)
    plt.axis('off')
    plt.title(f'Lables : {labels.tolist()}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()    

def print_summary(clean_acc, asr):
    """
    print evaluation summary
    Args:
        clean_acc : Clean accuracy
        asr : Attack success rate
    """ 
    print('\n', '=' * 50)
    print('Evaluation Summary : ')
    print(f'Clean Accuracy : {clean_acc:.2f}%')
    print(f'Attack Success Rate: {asr:.2f}%')
    print('='*50 + '\n')           
