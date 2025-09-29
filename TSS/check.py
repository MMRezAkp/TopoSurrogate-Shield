# In your debug script or __getitem__ in data.py
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from data import *
from model import *

def debug_visualize_trigger(save_path="trigger_visual_debug.png"):
    dataset = PoisenedCIFAR10(train=False, poison_ratio=1.0, target_label=0)
    img, label = dataset[0]
    img_un = TF.normalize(img, mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                                std=[1/0.2023, 1/0.1994, 1/0.2010])  # unnormalize

    plt.imshow(img_un.permute(1, 2, 0).clip(0, 1))
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved trigger visualization to {os.path.abspath(save_path)}")

debug_visualize_trigger()  


# Note: This script can work with any supported architecture. Change the architecture parameter as needed.
# Supported architectures: resnet18, resnet34, resnet152, mobilenet_v2, efficientnet_b0
model = get_model(num_classes=10, architecture='resnet18')
model.load_state_dict(torch.load("D:/toap_env/Backdoored_ResNet18/experiment-output/models/backdoored.pth"))
model.eval()

# Run on a batch of 100% poisoned data
_, _, poisoned_loader = get_data_loader(data_dir='./data', batch_size=64,
                                        poison_ratio=1.0, target_label=0,
                                        trigger_size=3, trigger_color=[1, 1, 1], train=False)

# Evaluate
correct, total = 0, 0
for images, _ in poisoned_loader:
    outputs = model(images)
    pred = outputs.argmax(dim=1)
    correct += (pred == 0).sum().item()  # your target label
    total += outputs.size(0)
print("Quick ASR:", correct / total * 100)