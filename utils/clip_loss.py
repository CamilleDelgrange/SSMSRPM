'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/utils/clip_loss.py
'''


from typing import Tuple, List

import torch
from torch import nn

class CLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5,
               quantile: float = 0.95,  # Threshold for what counts as a "hard negative"
               hard_negative_weight: float = 1.2, 
               smoothing_factor: float = 0.1,
               hard_neg: bool = False) -> None:
    super(CLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

    self.quantile = quantile  # Quantile for selecting hard negatives (e.g., 0.8 for top 20%)
    self.hard_negative_weight = hard_negative_weight  # Weight multiplier for hard negatives
    self.hard_neg = hard_neg
    self.smoothing_factor = smoothing_factor

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
    logits = torch.matmul(out0, out1.T) / self.temperature
    print("Logits", logits)
    labels = torch.arange(len(out0), device=out0.device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1

    if self.hard_neg:
      # Hard Negative Mining: Penalize high similarity negative pairs more
      with torch.no_grad():
        # Mask to identify off-diagonal elements (i.e., negatives)
        neg_mask = torch.ones_like(logits).fill_diagonal_(0)
        # Extract only the off-diagonal elements (negative pairs)
        off_diag_logits = logits[~torch.eye(logits.size(0), dtype=bool, device=logits.device)].view(logits.size(0), -1)

        # Sort logits and get the quantile threshold (e.g., 80th percentile for hard negatives)
        sorted_neg_logits, _ = torch.sort(off_diag_logits.view(-1))
        quantile_index = int(self.quantile * len(sorted_neg_logits))
        quantile_threshold = sorted_neg_logits[quantile_index].item()
        print("Quantile threshold", quantile_threshold)

        # Identify hard negatives using the quantile-based threshold
        hard_negatives_mask = (logits > quantile_threshold) * neg_mask

      # Apply the hard negative penalty
      if hard_negatives_mask.sum() > 0:
        hard_negatives = logits * hard_negatives_mask
        hard_negative_penalty = self.hard_negative_weight * hard_negatives.mean()
        loss += hard_negative_penalty
        print("Hard negatives", hard_negatives)
        print("Loss", loss)
  
    return loss, logits, labels