import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def save_cifar10_batch(output_path='input_images.pt', batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    images, _ = next(iter(testloader))  # Just get the first batch
    torch.save(images, output_path)
    print(f"Saved {batch_size} CIFAR-10 test images to {output_path}")

if __name__ == '__main__':
    save_cifar10_batch('input_images.pt')
