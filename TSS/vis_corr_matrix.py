import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to your correlation matrices
paths = {
    'Pearson': 'corr_matrix.npy',
    'Cosine': 'corr_matrix_cosine.npy'
}

for name, path in paths.items():
    if not os.path.exists(path):
        print(f"Skipping {name}: {path} not found.")
        continue

    mat = np.load(path)

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, cmap='viridis', square=True, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similarity'})
    plt.title(f'{name} Correlation Matrix Heatmap')
    plt.savefig(f'{name.lower()}_heatmap.png')
    plt.close()

    # Histogram
    plt.figure(figsize=(6, 4))
    values = mat[np.triu_indices_from(mat, k=1)]  # Upper triangle only (excluding diagonal)
    plt.hist(values, bins=50, color='gray', edgecolor='black')
    plt.title(f'{name} Correlation Histogram')
    plt.xlabel('Similarity Value')
    plt.ylabel('Frequency')
    plt.savefig(f'{name.lower()}_histogram.png')
    plt.close()

    print(f"Saved {name} heatmap and histogram as PNG.")