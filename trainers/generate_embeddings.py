'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/trainers/generate_embeddings.py
'''

import os 
import sys
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# Append the path to the root of the repository to sys.path
REPO_PATH = os.path.abspath(os.path.join(CURRENT_PATH, '..'))
sys.path.append(REPO_PATH)

from utils.pieces import DotDict
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datasets.TabularDataset import TabularDataset
from models.TabularEmbeddingModel import TabularEmbeddingModel
from models.ResnetEmbeddingModel import ResnetEmbeddingModel
from utils.utils import transformations_eval
from captum.attr import IntegratedGradients
from monai.data import Dataset, DataLoader

def generate_embeddings(hparams):
  """
  Generates embeddings using trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
  else:
      raise RuntimeError("This is intended for GPU, but no CUDA device is available")
  
  pl.seed_everything(hparams.seed)
  
  if hparams.eval_datatype == 'imaging' or hparams.eval_datatype == 'multimodal':
    val_image_paths = torch.load(hparams.data_val_imaging)
    val_labels = torch.load(hparams.labels_val)
    _, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    val_data = [{"image": str(path), "label": int(label)} for path, label in zip(val_image_paths, val_labels)]
    val_dataset = Dataset(data=val_data, transform=val_transforms)
    model = ResnetEmbeddingModel(hparams)

  elif hparams.eval_datatype == 'tabular':
    val_dataset = TabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = val_dataset.get_input_size()
    model = TabularEmbeddingModel(hparams)
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  drop = ((len(val_dataset)%hparams.batch_size)==1)
  # For integrated_gradients, put batch_size to 1!
  if hparams.integrated_gradients:
    hparams.batch_size = 1

  test_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  model.eval()
  if hparams.integrated_gradients:
    all_feature_importances = []
    with open(hparams.columns_name, 'r') as file:
      feature_names = [line.strip() for line in file]
    print(feature_names)
    print(f"Number of features: {len(feature_names)}")
    field_lengths=torch.load(hparams.field_lengths_tabular)
    i=0
    for batch in test_loader:
      batch_input = batch['image']
      print("Batch input", i, batch_input.shape)
      feature_importances = compute_integrated_gradients(batch_input, model, feature_names, field_lengths)
      all_feature_importances.append(feature_importances)
      i+=1

    if hparams.integrated_gradients:
      # Aggregate all feature importances
      all_feature_importances = pd.concat(all_feature_importances, axis=0).groupby('Feature').mean().reset_index()
      save_path = os.path.join(grab_rundir_from_checkpoint(hparams.checkpoint), 'ALL_feature_importances.csv')
      all_feature_importances.to_csv(save_path, index=False)
      print(f"Feature importances saved to {save_path}")
      feat_imp = pd.read_csv(f'{REPO_PATH}/runs/multimodal/best_model_name/ALL_feature_importances.csv')
      print(feat_imp)
      plot_feature_importances(feat_imp, f'{REPO_PATH}/runs/multimodal/best_model_name/ALL_feature_importance.png')
    
  # Save embeddings across val samples:
  for (loader, split) in [(test_loader, 'test')]:
    embeddings = []
    for batch in loader:
      batch_embeddings = model(batch['image']).detach()
      embeddings.extend(batch_embeddings)
    embeddings = torch.stack(embeddings)
    save_path = os.path.join(grab_rundir_from_checkpoint(hparams.checkpoint),f'{split}_embeddings_image_unimodal_pretrain_val.pt')
    torch.save(embeddings, save_path)

  test_embeddings = torch.load(save_path)
  print(f"Test embeddings shape: {test_embeddings.shape}")

def grab_rundir_from_checkpoint(checkpoint):
  return os.path.dirname(checkpoint)

def load_embeddings(file_path):
  """
  Loads embeddings from a specified file path.
  """
  return torch.load(file_path)

def compute_integrated_gradients(inputs, model, feature_names, field_lengths):
    """
    Computes integrated gradients for each dimension of the embedding outputs and aggregates
    the importance of each feature across all dimensions, specifically handling categorical features.

    Parameters:
    - inputs (Tensor): Input data for which attributions are to be calculated.
    - model (torch.nn.Module): The model to use for computing gradients.
    - feature_names (list of str): List of feature names.
    - field_lengths (Tensor): Tensor containing the number of choices for each categorical feature.

    Returns:
    - DataFrame: A DataFrame with averaged feature importances across all dimensions.
    """
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)  
    embedding_dim = model(inputs).shape[1]
    print(embedding_dim)
    all_attributions = []

    for dim in range(embedding_dim):
      attributions, delta = ig.attribute(inputs, baselines=baseline, target=dim, return_convergence_delta=True)
      attributions = attributions.abs()
      all_attributions.append(attributions)

    all_attributions = torch.stack(all_attributions, dim=-1) 
    mean_attributions = torch.mean(all_attributions, dim=-1) 

    summed_attributions = []
    start_idx = 0
    for length in field_lengths:
      end_idx = start_idx + length
      if length > 1:  
          summed_importance = mean_attributions[:, start_idx:end_idx].sum(dim=1)
      else:  
          summed_importance = mean_attributions[:, start_idx]
      summed_attributions.append(summed_importance)
      start_idx = end_idx

    if start_idx < mean_attributions.size(1):
      summed_attributions.extend(mean_attributions[:, start_idx:].unbind(dim=1))

    final_attributions = torch.cat([attr.unsqueeze(1) for attr in summed_attributions], dim=1)
    average_attributions = final_attributions.mean(dim=0)

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': average_attributions.detach().numpy()
    })

    return feature_importances

def plot_feature_importances(feature_importances, save_path):
  """
  Plots and saves feature importances.
  
  IN:
  feature_importances: DataFrame with feature importances
  morphometric_features: List of morphometric feature names
  save_path: Path to save the plot
  """
  # List of morphometric feature names
  morphometric_features = [
      'WMH', 'TV_WMH', 'TV_PV_WMH', 'volume_brain_GM_WM_norm', 'volume_GM_norm',
      'BMI_i2', 'waist_circumference_i2', 'height_i2', 'weight_i2'
  ]

  # Sort the feature importances in descending order and take the top 20
  feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

  # Determine the color based on whether the feature is morphometric or not
  colors = ['orange' if feature in morphometric_features else 'blue' for feature in feature_importances['Feature']]

  # Plot the feature importances
  plt.figure(figsize=(12, 6))
  plt.bar(feature_importances['Feature'], feature_importances['Importance'], color=colors)
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Top 20 Stroke Embedding Feature Importance by Integrated Gradients')
  plt.xticks(rotation=90)

  # Add legend with proper handles
  legend_handles = [
      Line2D([0], [0], color='orange', lw=4, label='Morphometric'),
      Line2D([0], [0], color='blue', lw=4, label='Non-Morphometric')
  ]
  plt.legend(handles=legend_handles)

  plt.tight_layout()
  plt.savefig(save_path)
  plt.show()
  print(f"Feature importances plot saved to {save_path}")

if __name__ == "__main__":
  args = DotDict({'eval_datatype' : 'tabular', 'data_test_eval_tabular': 'PATH_TO/features_test.csv',
  'labels_test_eval_tabular' : 'PATH_TO/labels_test.pt', 'eval_one_hot': True,
  'field_lengths_tabular' : 'PATH_TO/field_lengths_tabular.pt',
  'checkpoint': f'{REPO_PATH}/runs/multimodal/best_model_name/best_checkpoint.ckpt', 'embedding_dim': 2048,
  'model' : 'resnet50'})

  generate_embeddings(args)