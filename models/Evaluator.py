'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/models/Evaluator.py
'''

import sys
import os 
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# Append the path to the root of the repository to sys.path
REPO_PATH = os.path.abspath(os.path.join(CURRENT_PATH, '..'))
sys.path.append(REPO_PATH)

from typing import Dict
import json
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import numpy as np
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.MultimodalModelTabTransformer import MultimodalModelTransformer
from models.TabularModelTransformer import TabularModelTransformer
from models.utils.downstream_ensemble import BackboneEnsemble
from models.DAFT import DAFT
import wandb
import monai
from monai.transforms import ScaleIntensity

# Helper functions
def sum_heatmap_slices(gradcam_result):
  """
  Corrected helper function to sum GradCAM values for each slice in different planes.
  """
  # Sagittal (sum along y and z, keeping x slices)
  sagittal_sums = np.sum(gradcam_result, axis=(1, 2))  # Sum along height (y) and depth (z), keeping width (x)
  # Coronal (sum along x and z, keeping y slices)
  coronal_sums = np.sum(gradcam_result, axis=(0, 2))  # Sum along width (x) and depth (z), keeping height (y)
  # Axial (sum along x and y, keeping z slices)
  axial_sums = np.sum(gradcam_result, axis=(0, 1))  # Sum along width (x) and height (y), keeping depth (z)
  # Check the shapes of the summed arrays
  print("Axial sum shape:", axial_sums.shape, axial_sums)  # Should be (128,)
  print("Sagittal sum shape:", sagittal_sums.shape, sagittal_sums)  # Should be (128,)
  print("Coronal sum shape:", coronal_sums.shape, coronal_sums)  # Should be (128,)
  return sagittal_sums, coronal_sums, axial_sums

def find_top_slice(slice_sums):
  """ Helper function to find top-k slices with highest GradCAM values. """
  return np.argmax(slice_sums) # Get top k slices with highest sums

def mask_background(volume, percentile=1):
    """
    Mask the background using a dynamic threshold based on the percentile of intensity.
    Args:
        volume: The input volume.
        percentile: The intensity percentile to consider as background.
    """
    threshold = np.percentile(volume, percentile)
    mask = volume > threshold
    masked_volume = volume * mask
    return masked_volume, mask

def apply_mask_to_gradcam(gradcam_result, mask):
    """
    Apply the background mask to the GradCAM result.
    Args:
        gradcam_result: The GradCAM activation map.
        mask: The binary mask representing the brain region.
    Returns:
        gradcam_result_masked: GradCAM with masked background.
    """
    gradcam_result_masked = gradcam_result * mask
    return gradcam_result_masked

def normalize_volume(volume):
  """ Normalize the volume by dividing by the maximum value. """
  max_value = np.max(volume)
  if max_value > 0:
      return volume / max_value
  else:
      return volume

def compute_75th_percentile(volume):
  """ Compute the 75th percentile value of the volume. """
  return np.percentile(volume, 75)

def threshold_volume(volume, percentile_value):
  """ Apply the percentile threshold to the volume. """
  return volume > percentile_value  # Return a binary volume

def find_best_slice(thresholded_volume):
  """ Find the slice with the highest number of active voxels for each axis. """
  # Axial: slices along the z-axis
  axial_sums = np.sum(thresholded_volume, axis=(0, 1))  # Summing along x and y
  axial_slice = np.argmax(axial_sums)

  # Sagittal: slices along the x-axis
  sagittal_sums = np.sum(thresholded_volume, axis=(1, 2))  # Summing along y and z
  sagittal_slice = np.argmax(sagittal_sums)

  # Coronal: slices along the y-axis
  coronal_sums = np.sum(thresholded_volume, axis=(0, 2))  # Summing along x and z
  coronal_slice = np.argmax(coronal_sums)

  return axial_slice, sagittal_slice, coronal_slice

def custom_normalizer(x):
  def _compute(data):
    scaler = ScaleIntensity(minv=0.0, maxv=1.0)
    return np.stack([scaler(i) for i in data], axis=0)
    if isinstance(x, torch.Tensor):
      return torch.as_tensor(_compute(x.detach().cpu().numpy()), device=x.device)
  return _compute(x)


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.eval_datatype == 'imaging':
      self.model = ImagingModel(self.hparams)
    elif self.hparams.eval_datatype == 'tabular':
      if self.hparams.algorithm_name == 'TIP':
        if self.hparams.strategy == 'tip':
          self.model = TabularModelTransformer(self.hparams)
      else:
        self.model = TabularModel(self.hparams)
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
      if self.hparams.algorithm_name == 'SSMSRPM':
        assert self.hparams.strategy == 'ssmsrpm'
        if self.hparams.finetune_ensemble == True:
          self.model = BackboneEnsemble(self.hparams)
      elif self.hparams.algorithm_name == 'DAFT':
        self.model = DAFT(self.hparams)
      elif self.hparams.algorithm_name in set(['CONCAT']):
        if self.hparams.strategy == 'tip':
          self.model = MultimodalModelTransformer(self.hparams)
      elif self.hparams.algorithm_name == 'EVAL_PRETRAIN':
        # use MLP-based tabular encoder
        self.model = MultimodalModel(self.hparams)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    
    self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.cmat_val = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes)
    self.cmat_test = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes)
    self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.f1_test = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
    self.tpr_test = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
    
    self.criterion = torch.nn.CrossEntropyLoss()
    # for logging best so far validation AUC:
    self.best_val_score = 0 
    self.best_bacc_val = torchmetrics.MaxMetric()

    # To store samples corresponding to TP, FP, TN, FN:
    self.true_positive_images = []
    self.true_negative_images = []
    self.false_positive_images = []
    self.false_negative_images = []

    # Initialize the cumulative count to keep track of global indices
    self.cumulative_sample_count = 0

    # To store predictions and targets for calibration plot:
    self.test_predictions = []
    self.test_targets = []

    # To store val predictions for Youden index computation:
    self.val_predictions = []
    self.val_targets = []
    self.optimal_threshold = 0.0

    #print(self.model)

  def forward(self, x_im: torch.Tensor, x_tab:torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from data points. Adjust for different data types.
    """
    if self.hparams.eval_datatype == 'imaging':
        y_hat = self.model(x_im)
        #print("Prediction after forward", y_hat)
    elif self.hparams.eval_datatype == 'tabular':
        y_hat = self.model(x_tab)
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
        y_hat = self.model(x_im, x_tab)
    else:
        raise ValueError("Invalid datatype specified.")
    
    if len(y_hat.shape) == 1:
        y_hat = torch.unsqueeze(y_hat, 0)

    return y_hat

  def on_test_start(self):
    if self.hparams.youden_index:
      # Load the optimal threshold before testing
      threshold_load_path = f'{self.hparams.fig_dir}/optimal_threshold.json'
      with open(threshold_load_path, 'r') as f:
        data = json.load(f)
        self.optimal_threshold = data["optimal_threshold"]
      print(f"Optimal threshold loaded: {self.optimal_threshold}")
      # Initialize the confusion matrix for test
      if self.optimal_threshold != 0.0:
        print("Metrics with optimal threshold (YI)")
        task = 'binary'
        self.cmat_test = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes, threshold=self.optimal_threshold).to(self.device)  
        self.f1_test = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes, threshold=self.optimal_threshold).to(self.device)        
        self.tpr_test = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes, threshold=self.optimal_threshold).to(self.device)

  def test_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
    """
    Runs test step
    """
    x_im = batch.get('image', None)
    x_tab = batch.get('tab', None)
    y = batch['label']
    y_hat = self.forward(x_im, x_tab)
    print(f"Processing batch {batch_idx} with {len(y)} samples")
    print("Test logits after forward", y_hat)
    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes == 2:
        y_hat = y_hat[:, 1]
    # Store raw predictions and targets for calibration plot
    self.test_predictions.append(y_hat.detach().cpu())
    self.test_targets.append(y.detach().cpu())
    #self.acc_test(y_hat, y)
    self.auc_test(y_hat, y) #removed the update part temporarily
    self.cmat_test(y_hat, y)
    self.f1_test(y_hat, y)
    self.tpr_test(y_hat, y)

    # Compare y_hat to 0.5 to get predicted labels
    if self.hparams.youden_index:
      y_pred_labels = (y_hat >= self.optimal_threshold).int()
    else:
      y_pred_labels = (y_hat >= 0.5).int()
    
     # Identify true positive, true negative, false positive, and false negative indices within the batch
    true_positive_indices = ((y_pred_labels == 1) & (y == 1)).nonzero(as_tuple=True)[0]
    true_negative_indices = ((y_pred_labels == 0) & (y == 0)).nonzero(as_tuple=True)[0]
    false_positive_indices = ((y_pred_labels == 1) & (y == 0)).nonzero(as_tuple=True)[0]
    false_negative_indices = ((y_pred_labels == 0) & (y == 1)).nonzero(as_tuple=True)[0]

    # Calculate global indices for each sample in the batch
    global_indices = [self.cumulative_sample_count + i for i in range(len(y))]

    # Update the top 2 examples for each category based on the highest probabilities
    self.update_top_images(self.true_positive_images, true_positive_indices, x_im, y, y_pred_labels, y_hat, global_indices)
    self.update_top_images(self.true_negative_images, true_negative_indices, x_im, y, y_pred_labels, y_hat, global_indices)
    self.update_top_images(self.false_positive_images, false_positive_indices, x_im, y, y_pred_labels, y_hat, global_indices)
    self.update_top_images(self.false_negative_images, false_negative_indices, x_im, y, y_pred_labels, y_hat, global_indices)
    # Update the cumulative count after processing the batch
    self.cumulative_sample_count += len(y)
  
  def update_top_images(self, image_list, indices, images, true_labels, pred_labels, probs, global_indices):
    """
      Updates the list of top images with highest predicted probabilities for a given category.
      Keeps only the top two images with the highest probabilities.
    """
    for idx in indices:
      global_idx = global_indices[idx.item()]  # Retrieve the global index of the image
      # Handle the case when images are None
      if images is None:
        # If no image is available, store only the labels, probabilities, and global index
        current_image = (None, true_labels[idx], pred_labels[idx], probs[idx], global_idx)
      else: 
        current_image = (images[idx], true_labels[idx], pred_labels[idx], probs[idx], global_idx)  # Store image and global index
      image_list.append(current_image)
      image_list.sort(key=lambda x: x[3], reverse=True)
      # Keep only the top 2 images
      if len(image_list) > 50:
        image_list.pop()

  def test_epoch_end(self, outputs) -> None:
    """
    Test epoch end
    """
    #print(f"Processed {len(outputs)} batches in total")  # Debug statement
    test_auc = self.auc_test.compute()
    self.log('test.auc', test_auc)
    test_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.cmat_test)
    self.log("test.bacc", test_bacc)
    test_f1 = self.f1_test.compute()
    self.log('test.f1', test_f1)
    test_recall = self.tpr_test.compute()
    self.log('test.recall', test_recall)

    # Concatenate stored predictions and targets
    all_preds = torch.cat(self.test_predictions)
    all_targets = torch.cat(self.test_targets)
    
    self.plot_and_save_metrics(self.auc_test, self.cmat_test, 'test', all_preds, all_targets)

    if self.hparams.gradcam:
      # Compute GradCAM for the selected samples:
      # After collecting the top images, process them for GradCAM visualization
      output_path = f'{self.hparams.fig_dir}/GradCAM_quantitative_cases_new'
      
      # Concatenate the top images
      imgs = [img[0] for img in self.true_positive_images + self.true_negative_images + self.false_positive_images + self.false_negative_images]
      true_labels = [img[1].item() for img in self.true_positive_images + self.true_negative_images + self.false_positive_images + self.false_negative_images]
      pred_probs = [img[3].item() for img in self.true_positive_images + self.true_negative_images + self.false_positive_images + self.false_negative_images]
      pred_labels = [img[2].item() for img in self.true_positive_images + self.true_negative_images + self.false_positive_images + self.false_negative_images]
    
      #self._process_samples(imgs, true_labels, pred_probs, pred_labels, output_path)
      self.plot_patients_slices(imgs, true_labels, pred_probs, pred_labels, output_path)
      # At the end of the test loop, you can inspect the global indices in the image lists
      print("True Positive Images and Global Indices:", [(x[4], x[3]) for x in self.true_positive_images])  # Prints global indices and probabilities
      print("True Negative Images and Global Indices:", [(x[4], x[3]) for x in self.true_negative_images])
      print("False Positive Images and Global Indices:", [(x[4], x[3]) for x in self.false_positive_images])
      print("False Negative Images and Global Indices:", [(x[4], x[3]) for x in self.false_negative_images])
    
    # reset metrics at the end of every epoch
    self.auc_test.reset()
    self.cmat_test.reset()
    self.f1_test.reset()
    self.tpr_test.reset()

  def _process_samples(self, imgs, true_labels, pred_probs, pred_labels, output_path):
    """
    Generate and plot selected slices with GradCAM overlay.
    """
    def custom_normalizer(x):
      def _compute(data):
        scaler = ScaleIntensity(minv=0.0, maxv=1.0)
        return np.stack([scaler(i) for i in data], axis=0)
        if isinstance(x, torch.Tensor):
          return torch.as_tensor(_compute(x.detach().cpu().numpy()), device=x.device)
      return _compute(x)

    # Set the model to evaluation mode and move it to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.hparams.eval_datatype == 'imaging':
      imaging_model = self.model
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
      imaging_model = self.model.imaging_model
    imaging_model.to(device)
    # Ensure the model is in evaluation mode but allow gradients for GradCAM
    imaging_model.eval()
    #for name, _ in self.model.named_modules(): print("Layers", name)

    n_examples = len(imgs)
    print(n_examples)
    print(len(imgs), imgs[0].shape, true_labels, pred_probs, pred_labels)
    subplot_shape = [2, n_examples]  # 2 rows (original + overlay), n_examples columns
    fig, axes = plt.subplots(*subplot_shape, figsize=(25, 10), facecolor="white")
    
    cam = monai.visualize.GradCAM(nn_module=imaging_model, target_layers="encoder.7", postprocessing=custom_normalizer)  # Adjust target layer as needed
    for name, _ in imaging_model.named_modules(): print("Layers", name)
    for i in range(n_examples):
      img = imgs[i].unsqueeze(0).to(device)  # Move image to the correct device and add batch dimension
      # Ensure the input tensor requires gradients
      img.requires_grad = True
      #print(img.shape)
      true_label = true_labels[i]
      pred_label = pred_labels[i]
      #print(pred_label)
      pred_prob = pred_probs[i]

      # Set the middle slice for 3D image
      the_slice = img.shape[-1] // 2  # Assuming 3D images
      #img = img.requires_grad_()  # Enforce gradient calculation for input image
      # Generate the CAM result for the predicted class and enable the gradients that are disabled by default in pytorch lightning!
      with torch.set_grad_enabled(True):
        cam_result = cam(x=img, class_idx=pred_label)
        cam_result = cam_result[..., the_slice]  # Extract the same slice for CAM result
      if isinstance(cam_result, torch.Tensor):
        cam_result = cam_result.cpu().detach().numpy()

      img_slice = img[..., the_slice].detach().cpu().numpy()  # Get the middle slice

      # Plot the original image slice
      ax = axes[0, i]
      ax.imshow(img_slice[0][0], cmap="gray")
      ax.set_title(f"True: {true_label}\nPred: {pred_label}\nProba: {pred_prob:.2f}", fontsize=15)
      ax.axis("off")

      # Plot the GradCAM overlay
      ax = axes[1, i]
      ax.imshow(img_slice[0][0], cmap="gray")  # Original image in grayscale
      ax.imshow(cam_result[0][0], cmap="jet", alpha=0.5)  # GradCAM heatmap overlay
      ax.set_title("GradCAM Overlay", fontsize=15)
      ax.axis("off")

    # Save the figure with all subplots to the specified path
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

  def plot_patients_slices(self, imgs, true_labels, pred_probs, pred_labels, output_path):
    """
    Plot the most informative slices for each patient in axial, sagittal, and coronal views.
    Each row corresponds to a different view, and each column corresponds to a different patient.
    
    Args:
      imgs: List of 3D input volumes for each patient.
      true_labels: List of true labels for each patient.
      pred_probs: List of predicted probabilities for each patient.
      pred_labels: List of predicted labels for each patient.
      output_path: Output viz
    """
    activations_healthy = []
    activations_stroke = []

    # Categorize patients into TP, TN, FP, FN
    TP_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == 1 and pred == 1 and imgs[i] is not None]
    TN_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == 0 and pred == 0 and imgs[i] is not None]
    FP_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == 0 and pred == 1 and imgs[i] is not None]
    FN_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == 1 and pred == 0 and imgs[i] is not None]

    # Select up to 10 samples from each category
    TP_samples = TP_indices[:10]
    TN_samples = TN_indices[:10]
    FP_samples = FP_indices[:10]
    FN_samples = FN_indices[:10]

    # Group samples and label categories
    all_samples = [TP_samples, TN_samples, FP_samples, FN_samples]
    categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

    # Set the model to evaluation mode and move it to the appropriate device (e.g., GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.hparams.eval_datatype == 'imaging':
      imaging_model = self.model
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
      imaging_model = self.model.imaging_model
    imaging_model.to(device)
    # Ensure the model is in evaluation mode but allow gradients for GradCAM
    imaging_model.eval()
    #for name, _ in self.model.named_modules(): print("Layers", name)

    print(len(imgs), imgs[0].shape, true_labels, pred_probs, pred_labels)
    gradcam = monai.visualize.GradCAM(nn_module=imaging_model, target_layers="encoder.7", postprocessing=custom_normalizer)  # Adjust target layer as needed
    
    for category, sample_indices in zip(categories, all_samples):
      n_patients = len(sample_indices)
      if n_patients == 0:
        continue 

      fig, axes = plt.subplots(3, n_patients, figsize=(n_patients * 3, 9), facecolor="white")
      plt.subplots_adjust(wspace=0.05, hspace=0.3)

      for i, patient_idx in enumerate(sample_indices):
        true_label = true_labels[patient_idx]
        pred_label = pred_labels[patient_idx]
        pred_prob = pred_probs[patient_idx]
        img = imgs[patient_idx]
    
        if isinstance(img, torch.Tensor):
          img = img.cpu().detach().numpy()

        # Mask bg:
        masked_volume, mask = mask_background(img)
        # normalize
        normalized_volume = normalize_volume(masked_volume)
        # Convert to Tensor, add batch dim and move to device
        img_f = torch.tensor(normalized_volume).unsqueeze(0).to(device) 
        # Ensure the input tensor requires gradients
        img_f.requires_grad = True
        # Generate the CAM result for the predicted class and enable the gradients that are disabled by default in pytorch lightning!
        with torch.set_grad_enabled(True):
          gradcam_result = gradcam(x=img_f, class_idx=pred_label)
        
        # Extract tensors before plotting:
        if isinstance(gradcam_result, torch.Tensor):
          gradcam_result = gradcam_result.cpu().detach().numpy()
        if isinstance(img_f, torch.Tensor):
          img_f = img_f.cpu().detach().numpy()
        print("Activation map", gradcam_result.shape)

        if self.hparams.grad_cam_strategy == 'accumulation':
          # Find top informative slices based on GradCAM heatmap
          gradcam_result = apply_mask_to_gradcam(gradcam_result, mask)
          mean_activation_gradcam = gradcam_result.mean()
          if pred_label == 1: #Stroke patients
            activations_stroke.append(mean_activation_gradcam)
          else: #Healthy patients
            activations_healthy.append(mean_activation_gradcam)
          axial_sums, sagittal_sums, coronal_sums = sum_heatmap_slices(gradcam_result[0,0])
          axial_slice = find_top_slice(axial_sums)
          sagittal_slice = find_top_slice(sagittal_sums)
          coronal_slice = find_top_slice(coronal_sums)
          print("Axial slice", axial_slice)
          print("Coronal slice", coronal_slice)
          print("Sagittal slice", sagittal_slice)

        elif self.hparams.grad_cam_strategy == 'percentile':
          # compute 75th percentile threshold
          percentile_value = compute_75th_percentile(gradcam_result)
          thresholded_volume = threshold_volume(gradcam_result, percentile_value)
          # Find most informative slices for axial, sagittal, and coronal views
          axial_slice, sagittal_slice, coronal_slice = find_best_slice(thresholded_volume)

        # Plot Axial Slice
        axes[0, i].imshow(np.rot90(img_f[0, 0, :, :, axial_slice]), cmap='gray')
        axes[0, i].imshow(np.rot90(gradcam_result[0, 0, :, :, axial_slice]), cmap='jet', alpha=0.5)
        axes[0, i].axis('off')
        
        # Plot Sagittal Slice
        axes[1, i].imshow(np.rot90(img_f[0, 0, sagittal_slice, :, :]), cmap='gray')
        axes[1, i].imshow(np.rot90(gradcam_result[0, 0, sagittal_slice, :, :]), cmap='jet', alpha=0.5)
        axes[1, i].axis('off')

        # Plot Coronal Slice
        axes[2, i].imshow(np.rot90(img_f[0, 0, :, coronal_slice, :]), cmap='gray')
        axes[2, i].imshow(np.rot90(gradcam_result[0, 0, :, coronal_slice, :]), cmap='jet', alpha=0.5)
        axes[2, i].axis('off')
    
        # Add titles with true and predicted labels + probabilities
        axes[0, i].set_title(f'True: {true_labels[patient_idx]}\nPred: {pred_labels[patient_idx]}\nProba: {pred_probs[patient_idx]:.2f}',
                              fontsize=12, y=1.1)

      # Set row labels for the views
      axes[0, 0].set_ylabel('Axial View', fontsize=12)
      axes[1, 0].set_ylabel('Sagittal View', fontsize=12)
      axes[2, 0].set_ylabel('Coronal View', fontsize=12)
      
      # Save the figure with all subplots to the specified path
      fig.savefig(f'{output_path}_{category}_masked.png', bbox_inches="tight", dpi=150)
      plt.close(fig)
      print(f"Visualization for {category} saved to {output_path}_{category}.png")
      
      data = {
        'Activation': activations_stroke + activations_healthy,
        'Patient Type': ['Stroke'] * len(activations_stroke) + ['Healthy'] * len(activations_healthy)
      }

      # Create the boxplot
      plt.figure(figsize=(10, 6))
      sns.boxplot(x='Patient Type', y='Activation', data=data)
      plt.title('GradCAM Activation for Masked T2-FLAIR MRI')
      plt.xlabel('Patient Type')
      plt.ylabel('Mean Activation')
      plt.grid(True)
      plt.savefig(f"{output_path}_boxplot_masked.png", bbox_inches="tight", dpi=150)
      plt.show()

      print(f"Boxplot saved to {output_path}_boxplot_masked.png")

  def plot_and_save_metrics(self, auc_metric, cmat_metric, phase, all_preds, all_targets):
    # Compute ROC curve
    targets_cpu = all_targets.cpu().numpy()  # Move to CPU and convert to numpy
    preds_cpu = all_preds.cpu().numpy()  # Move to CPU and convert to numpy
    print("Preds cpu", preds_cpu)
    
    fpr, tpr, _ = roc_curve(targets_cpu, preds_cpu)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{phase} ROC curve (area = {auc_metric.compute():.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{phase} Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_curve_path = f'{self.hparams.fig_dir}/{phase}_roc_curve.png'
    plt.savefig(roc_curve_path)
    plt.close()
    
    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(targets_cpu, preds_cpu, n_bins=3)
    plt.figure()
    plt.plot(prob_pred, prob_true, 's-', label='Calibration plot')
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'{phase} Calibration Plot')
    plt.legend(loc='upper left')
    calibration_path = f'{self.hparams.fig_dir}/{phase}_calibration_plot.png'
    plt.savefig(calibration_path)
    plt.close()
    wandb.log({f"{phase}/calibration_plot": wandb.Image(calibration_path)})

    # Confusion matrix
    cmat = cmat_metric.compute().cpu().numpy()
    print("cmat test", cmat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
    disp.plot()
    plt.title(f'{phase} Confusion Matrix')
    confusion_matrix_path = f'{self.hparams.fig_dir}/{phase}_confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Additional WandB logging for ROC curve and confusion matrix
    if self.hparams.youden_index:
      if self.hparams.num_classes == 2:
        y_pred = (preds_cpu >= self.optimal_threshold).astype(int) #self.youden_index
    elif self.hparams.num_classes == 2:
      y_pred = (preds_cpu >= 0.5).astype(int) #self.youden_index
    else:
      y_pred = preds_cpu.argmax(axis=1)

     # Convert 1D array of class 1 probabilities to a 2D array [p_class0, p_class1]
    preds_2d = np.stack([1 - preds_cpu, preds_cpu], axis=1)
    # WandB ROC curve logging
    wandb.log({"roc": wandb.plot.roc_curve(targets_cpu, preds_2d, ['no_stroke', 'stroke'])})

    # WandB confusion matrix logging
    wandb.sklearn.plot_confusion_matrix(targets_cpu, y_pred)

  def log_correctly_classified_images(self):
    for img, true_label, pred in self.correct_classified_images:
      img_slice = self.get_middle_slice(img)
      fig, ax = plt.subplots()
      ax.imshow(img_slice, cmap='gray')
      ax.set_title(f"True: {true_label.item()}, Pred: {pred.item()}")
      ax.axis('off')
      # Log the figure to WandB
      self.logger.experiment.log({f"Correctly classified image": wandb.Image(fig, caption=f"True: {true_label.item()}, Pred: {pred.item():.2f}")})
      plt.close(fig)
  
  def _log_image_to_wandb(self, img, true_label, pred, description):
    img_slice = self.get_middle_slice(img)  # Assuming `get_middle_slice` is a method to extract the middle slice
    fig, ax = plt.subplots()
    ax.imshow(img_slice, cmap='gray')
    ax.set_title(f"True: {true_label.item()}, Pred: {pred.item()}")
    ax.axis('off')
    
    # Log the figure to WandB
    self.logger.experiment.log({
        f"{description}": wandb.Image(fig, caption=f"True: {true_label.item()}, Pred: {pred.item():.2f}")
    })
    plt.close(fig)

  def get_middle_slice(self, img):
    img = img.cpu().numpy()
    middle_slice = img[0, :, :, img.shape[3] // 2]
    return middle_slice

  def training_step(self, batch: Dict[str, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    x_im = batch.get('image', None)
    x_tab = batch.get('tab', None)
    y = batch['label']

    y_hat = self.forward(x_im, x_tab)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.auc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)

    # Log the learning rate
    optimizer = self.optimizers()
    lr = optimizer.param_groups[0]['lr']
    self.log('lr-AdamW', lr, on_epoch=True, on_step=True)  # Log every step
    
    return loss
  
  def on_train_start(self):
    # by default lightning executes validation step sanity checks before training starts,
    # so we need to make sure val_acc_best doesn't store accuracy from these checks
    self.best_bacc_val.reset()

  def training_epoch_end(self, _) -> None:
    """
    Compute training epoch metrics and check for new best values
    """
    self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

  def validation_step(self, batch: Dict[str, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x_im = batch.get('image', None)
    x_tab = batch.get('tab', None)
    y = batch['label']
    y_hat = self.forward(x_im, x_tab)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    if self.hparams.youden_index:
      self.val_predictions.append(y_hat.detach().cpu())
      self.val_targets.append(y.detach().cpu())

    self.auc_val(y_hat, y) 
    self.cmat_val(y_hat, y)

    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

  def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: torchmetrics.ConfusionMatrix):
    # Confusion matrix whose i-th row and j-th column entry indicates
    # the number of samples with true label being i-th class and
    # predicted label being j-th class.
    cmat = confusion_matrix.compute()
    # Compute the recall for each class (handle division by zero)
    recalls = cmat.diag() / cmat.sum(dim=1).clamp(min=1)
    
    # Compute the balanced accuracy
    balanced_accuracy = recalls.mean().item()
    
    return balanced_accuracy
    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_auc_val = self.auc_val.compute() 
    epoch_bacc_val = self._get_balanced_accuracy_from_confusion_matrix(self.cmat_val) 
    self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
    self.log("eval.val.bacc", epoch_bacc_val, on_epoch=True, on_step=False, metric_attribute=self.cmat_val, prog_bar=False)
    self.best_bacc_val(epoch_bacc_val)
    self.log("eval.val.best_bacc", self.best_bacc_val.compute(), on_epoch=True)

    if self.hparams.youden_index:
      all_preds = torch.cat(self.val_predictions)
      all_targets = torch.cat(self.val_targets)
      self.compute_youden_index(all_preds, all_targets)

    self.best_val_score = max(self.best_val_score, epoch_auc_val)
    self.log("eval.val.best_auc", self.best_val_score, on_epoch=True)

    self.auc_val.reset()
    self.cmat_val.reset()

  def compute_youden_index(self, all_preds, all_targets):
    fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(np.abs(youden_index))
    optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = float(thresholds[optimal_idx])  
    threshold_save_path = f'{self.hparams.fig_dir}/optimal_threshold.json'
    with open(threshold_save_path, 'w') as f:
        json.dump({"optimal_threshold": optimal_threshold}, f)
    print(f"Optimal threshold saved to {threshold_save_path}")
    self.log("eval.val.optimal_threshold", optimal_threshold, on_epoch=True)

          
  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    if self.hparams.optimizer_eval == 'adam':
      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    elif self.hparams.optimizer_eval == 'adamw':
      optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    else:
      raise ValueError('Valid schedulers are "adam" and "adamw"')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(2/self.hparams.check_val_every_n_epoch)) #min_lr=self.hparams.lr_eval*0.0001
        
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **kwargs):
    scaler = getattr(self.trainer.strategy.precision_plugin, "scaler", None)
    if scaler:
        scale_before_step = scaler.get_scale()
    optimizer.step(closure=optimizer_closure)
    if scaler:
        scale_after_step = scaler.get_scale()
        self.should_skip_lr_scheduler_step = scale_before_step > scale_after_step
    else:
        self.should_skip_lr_scheduler_step = False

  def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    if self.should_skip_lr_scheduler_step:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


if __name__ == "__main__":
  # Mock data creation
  def create_mock_data(num_patients=3, vol_shape=(2, 2, 2)): 
    imgs = [np.random.rand(*vol_shape) for _ in range(num_patients)]
    true_labels = [1, 0, 1]
    pred_probs = [0.85, 0.2, 0.75]
    pred_labels = [1, 0, 1]
    return imgs, true_labels, pred_probs, pred_labels

  # Generate mock data
  imgs, true_labels, pred_probs, pred_labels = create_mock_data()

  # Mock function for GradCAM without using GPU
  def mock_gradcam(img):
    # Simulate a GradCAM output (random values in the same shape as img)
    return np.random.rand(*img.shape)

  def plot_patients_slices(imgs, true_labels, pred_probs, pred_labels, output_path):
    """
    Plot the most informative slices for each patient in axial, sagittal, and coronal views.
    Each row corresponds to a different view, and each column corresponds to a different patient.
    
    Args:
      imgs: List of 3D input volumes for each patient.
      true_labels: List of true labels for each patient.
      pred_probs: List of predicted probabilities for each patient.
      pred_labels: List of predicted labels for each patient.
      output_path: Output viz
    """
    n_patients = len(imgs)
    fig, axes = plt.subplots(3, n_patients, figsize=(15, 10), facecolor="white")
    print(n_patients)
    print(len(imgs), imgs[0].shape, true_labels, pred_probs, pred_labels)

    for i in range(n_patients):
      true_label = true_labels[i]
      pred_label = pred_labels[i]
      pred_prob = pred_probs[i]
      #print(pred_label)
      img = imgs[i]
      print("Img values", img, img.shape)
      # Mask bg:
      masked_volume, mask = mask_background(img)
      print("Masked values", masked_volume)
      # normalize
      normalized_volume = normalize_volume(masked_volume)
      print("Img f", normalized_volume)
      # Mock GradCAM output for testing
      gradcam_result = mock_gradcam(normalized_volume)
      
      # Find top informative slices based on GradCAM heatmap
      axial_sums, sagittal_sums, coronal_sums = sum_heatmap_slices(gradcam_result)
      axial_slice = find_top_slice(axial_sums)
      sagittal_slice = find_top_slice(sagittal_sums)
      coronal_slice = find_top_slice(coronal_sums)
        
      # Plot Axial Slice
      axes[0, i].imshow(normalized_volume[:, :, axial_slice], cmap='gray')
      axes[0, i].imshow(gradcam_result[:, :, axial_slice], cmap='jet', alpha=0.5)
      axes[0, i].set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}\nProba: {pred_probs[i]:.2f}')
      axes[0, i].axis('off')
      
      # Plot Sagittal Slice
      axes[1, i].imshow(normalized_volume[sagittal_slice, :, :], cmap='gray')
      axes[1, i].imshow(gradcam_result[sagittal_slice, :, :], cmap='jet', alpha=0.5)
      axes[1, i].axis('off')

      # Plot Coronal Slice
      axes[2, i].imshow(normalized_volume[:, coronal_slice, :], cmap='gray')
      axes[2, i].imshow(gradcam_result[:, coronal_slice, :], cmap='jet', alpha=0.5)
      axes[2, i].axis('off')
  
    # Set row labels for the views
    axes[0, 0].set_ylabel('Axial View', fontsize=12)
    axes[1, 0].set_ylabel('Sagittal View', fontsize=12)
    axes[2, 0].set_ylabel('Coronal View', fontsize=12)
    
    # Save the figure with all subplots to the specified path
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

  output_path = 'PATH_TO_REPO/figures/GradCAMquant_test.png'
  plot_patients_slices(imgs, true_labels, pred_probs, pred_labels, output_path)