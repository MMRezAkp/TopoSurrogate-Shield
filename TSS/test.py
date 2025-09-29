import torch

act = torch.load('activation_output/poisoned_layer3_1.pt')

print("Shape:", act.shape)
print("Min:", act.min().item())
print("Max:", act.max().item())
print("Mean:", act.mean().item())
print("Std Dev:", act.std().item())