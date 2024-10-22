
import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.Tip_utils.Transformer import TabularTransformerEncoder

class MultimodalModelTransformer(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  For CONCAT method. Use TIP's transformer-based tabular encoder
  """
  def __init__(self, args) -> None:
    super(MultimodalModelTransformer, self).__init__()
    print('Use transformer for tabular data.')
    self.missing_tabular = args.missing_tabular
    print(f'Current missing tabular for TransformerTabularModel: {self.missing_tabular}')
    
    self.imaging_model = ImagingModel(args)
    self.strategy = args.strategy
    self.create_tabular_model(args)
    self.fusion_method = args.algorithm_name
    print('Fusion method: ', self.fusion_method)
    if self.fusion_method == 'CONCAT':
      in_dim = args.embedding_dim + args.tabular_embedding_dim
    else:
      raise ValueError('Fusion method not recognized.')
    self.head = nn.Linear(in_dim, args.num_classes)


  def create_tabular_model(self, args):
    self.field_lengths_tabular = torch.load(args.field_lengths_tabular)
    self.cat_lengths_tabular = []
    self.con_lengths_tabular = []
    for x in self.field_lengths_tabular:
      if x == 1:
        self.con_lengths_tabular.append(x) 
      else:
        self.cat_lengths_tabular.append(x)
    self.num_con = len(self.con_lengths_tabular)
    self.num_cat = len(self.cat_lengths_tabular)
    if self.strategy == 'tip':
      self.tabular_model = TabularTransformerEncoder(args, self.cat_lengths_tabular, self.con_lengths_tabular)
      print('Using TIP tabular encoder')
    else:
      assert False, f'Strategy not recognized: {self.strategy}'

  def forward(self, x_im: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_model.encoder(x_im).squeeze()
    if self.strategy == 'tip':
      if self.missing_tabular:
        missing_mask = x_tab[1]
        x_tab = self.tabular_model(x=x_tab, mask=missing_mask, mask_special=missing_mask)[:,0,:]
      else:
        x_tab = self.tabular_model(x_tab)[:,0,:]

    if self.fusion_method == 'CONCAT':
      x = torch.cat([x_im, x_tab], dim=1)
    x = self.head(x)
    return x
