'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
'''

"""
UMAP Visualization for Unimodal and Cross-modal Embeddings

This script generates UMAP plots to visualize the latent space distribution of embeddings from both unimodal and cross-modal models.

The script accepts embedding files for each model type (imaging and tabular data, both unimodal and multimodal) and projects them 
into a 2D space using the UMAP algorithm. The resulting plots provide insights into how well-separated or clustered the embeddings are, 
depending on the pretraining method. This can help assess the quality of feature learning in different models.

Key Features:
- Supports visualization of both imaging and tabular embeddings.
- Compares unimodal embeddings (pretrained separately) with cross-modal embeddings (trained jointly).
- Saves the resulting UMAP plots in a specified directory for further analysis.

Usage:
Ensure that the correct paths to the embedding files are set in the script. After execution, UMAP plots will be saved in the defined 
output directory.
"""

import umap
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(CURRENT_PATH, 'embeddings_vis')

paths = {
    'imaging_unimodal': os.path.join(CURRENT_PATH, 'PATH_TO_REPO/runs/imaging/jumping-donkey-183/test_embeddings_image_unimodal_pretrain_val.pt'),
    'imaging_multimodal': os.path.join(CURRENT_PATH, 'PATH_TO_REPO/runs/multimodal/gentle-sound-239/test_embeddings_image_multimodal.pt'),
    'tabular_multimodal': os.path.join(CURRENT_PATH, 'PATH_TO_REPO/runs/multimodal/gentle-sound-239/test_embeddings_tabular_multimodal.pt'),
    'tabular_unimodal': os.path.join(CURRENT_PATH, 'PATH_TO_REPO/runs/tabular/comfy-dust-243/test_embeddings_tabular_unimodal_pretrain_val.pt'),
    'labels': os.path.join(CURRENT_PATH, 'PATH_TO_LABELS/labels_pretrain_val_all_patients.pt')
}

os.makedirs(save_dir, exist_ok=True)

embeddings = {key: torch.load(path) for key, path in paths.items()}
labels = torch.load(paths['labels'])
labels = np.array(labels)

def plot_umap(embeddings_tab, embeddings_img, labels, title="UMAP", output_path=None):
    reducer_tab = umap.UMAP(random_state=42)
    reducer_img = umap.UMAP(random_state=42)
    
    umap_tab = reducer_tab.fit_transform(embeddings_tab)
    umap_img = reducer_img.fit_transform(embeddings_img)
    
    plt.figure(figsize=(10, 7))
    
    plt.scatter(umap_img[labels == 0, 0], umap_img[labels == 0, 1], c='blue', alpha=0.7, marker='s', label='Control Imaging')
    plt.scatter(umap_img[labels == 1, 0], umap_img[labels == 1, 1], c='blue', alpha=0.7, marker='o', label='Stroke Imaging')
    
    plt.scatter(umap_tab[labels == 0, 0], umap_tab[labels == 0, 1], c='red', alpha=0.7, marker='s', label='Control Tabular')
    plt.scatter(umap_tab[labels == 1, 0], umap_tab[labels == 1, 1], c='red', alpha=0.7, marker='o', label='Stroke Tabular')
    
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Control Imaging', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Stroke Imaging', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Control Tabular', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Stroke Tabular', markerfacecolor='red', markersize=10)
    ]

    plt.legend(handles=legend_elements, title="Diagnosis and Modality")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Save the plot if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    plt.show()

plot_umap(embeddings['tabular_unimodal'], embeddings['imaging_unimodal'], labels,
          title="UMAP - Unimodal Tabular vs Imaging",
          output_path=os.path.join(save_dir, 'umap_unimodal_tab_vs_imaging_labels_pretrain_val.png'))

plot_umap(embeddings['tabular_multimodal'], embeddings['imaging_multimodal'], labels,
          title="UMAP - Multimodal Tabular vs Imaging",
          output_path=os.path.join(save_dir, 'umap_multimodal_tab_vs_imaging_labels_pretrain_val.png'))

def compute_cosine_similarity(tab_embeddings, img_embeddings):
    return cosine_similarity(tab_embeddings, img_embeddings)

def top_k_accuracy(similarity_matrix, k_values):
    n_samples = similarity_matrix.shape[0]
    accuracies = []
    
    for k in k_values:
        correct_matches = 0
        for i in range(n_samples):
            sorted_indices = np.argsort(-similarity_matrix[i])
            
            if i in sorted_indices[:k]:
                correct_matches += 1
                
        accuracy = correct_matches / n_samples
        accuracies.append(accuracy)
    
    return accuracies

def plot_corrected_top_k_accuracy(unimodal_tab_embeddings, unimodal_img_embeddings, 
                                  multimodal_tab_embeddings, multimodal_img_embeddings,
                                  k_values, output_path=None):
    similarity_matrix_unimodal = cosine_similarity(unimodal_tab_embeddings, unimodal_img_embeddings)
    diagonal_scores_unimodal = np.diag(similarity_matrix_unimodal)

    print("Diagonal scores unimodal (cosine similarity of corresponding tabular and MRI pairs):")
    print(diagonal_scores_unimodal)  
    similarity_matrix_multimodal = cosine_similarity(multimodal_tab_embeddings, multimodal_img_embeddings)
    diagonal_scores_multimodal = np.diag(similarity_matrix_multimodal)

    print("Diagonal scores multimodal (cosine similarity of corresponding tabular and MRI pairs):")
    print(diagonal_scores_multimodal)   
    accuracies_unimodal = top_k_accuracy(similarity_matrix_unimodal, k_values)
    
    accuracies_multimodal = top_k_accuracy(similarity_matrix_multimodal, k_values)
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(k_values, accuracies_unimodal, label='Unimodal', linestyle='--', marker='s', color='red')
    plt.plot(k_values, accuracies_multimodal, label='Cross-modal', linestyle='--', marker='o', color='green')
    
    random_baseline = [k / max(k_values) for k in k_values]
    plt.plot(k_values, random_baseline, label='Random Baseline', linestyle='-', marker='x', color='black')
    
    plt.xlabel('Top k')
    plt.ylabel('Accuracy (Correct Matches in Top K)')
    plt.title('Corrected Top-K Accuracy for Unimodal vs Multimodal Embeddings')
    plt.legend()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show()

k_values = np.arange(1, 93, 1) 

plot_corrected_top_k_accuracy(embeddings['tabular_unimodal'], embeddings['imaging_unimodal'],
                              embeddings['tabular_multimodal'], embeddings['imaging_multimodal'],
                              k_values, output_path='/cluster/home/cdelgrange/MMCL/embeddings_vis/corrected_top_k_accuracy_unimodal_vs_multimodal.png')