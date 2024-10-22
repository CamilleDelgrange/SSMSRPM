'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/models/MultimodalModel.py
'''

import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from utils.pieces import DotDict


class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)
    if args.model=='resnet50':
      in_dim = 4096  # This assumes concatenated dimension is 4096
    else:
      in_dim = 1024
    self.head = nn.Linear(in_dim, args.num_classes)

  def forward(self, x_im: torch.Tensor, x_tab=None) -> torch.Tensor:
    #print("Shape before concat", x_im.shape)
    x_im = self.imaging_model.encoder(x_im).squeeze() #x_im.contiguous()
    x_tab = self.tabular_model.encoder(x_tab).squeeze()
    
    # Ensure the output is flattened if not already
    if len(x_im.shape) > 2:
        x_im = x_im.view(x_im.size(0), -1)
    if len(x_tab.shape) > 2:
        x_tab = x_tab.view(x_tab.size(0), -1)
    
    #print(f"x_im shape: {x_im.shape}")
    #print(f"x_tab shape: {x_tab.shape}")
    
    x = torch.cat([x_im, x_tab], dim=1)
    
    #print(f"Concatenated x shape: {x.shape}")
    
    x = self.head(x)
    return x

if __name__ == "__main__":
  args = DotDict({})
  model = MultimodalModel(args)
  x_im = torch.randn(6, 512)  # Example image tensor
  x_tab = torch.randn(6, 512)  # Example tabular tensor
  output = model(x_im, x_tab)
  print(output)