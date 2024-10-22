'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/models/TabularModel.py
'''

import torch
import torch.nn as nn
from collections import OrderedDict

from models.Tip_utils.Transformer import TabularTransformerEncoder

class TabularModelTransformer(nn.Module):
    """
    Evaluation model for tabular data only using TIP's transformer-based tabular encoder.
    This is for evaluating the tabular transformer on a classification task.
    """
    def __init__(self, args) -> None:
        super(TabularModelTransformer, self).__init__()
        print('Using transformer for tabular data.')
        self.missing_tabular = args.missing_tabular
        print(f'Current missing tabular for TransformerTabularModel: {self.missing_tabular}')
        
        # Create tabular model
        self.strategy = args.strategy
        self.create_tabular_model(args)
        
        # Set up the head for classification
        in_dim = args.tabular_embedding_dim
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
        
        # Initialize Tabular Transformer Encoder
        if self.strategy == 'tip':
            self.tabular_model = TabularTransformerEncoder(args, self.cat_lengths_tabular, self.con_lengths_tabular)
            print('Using TIP tabular encoder')
        else:
            assert False, f'Strategy not recognized: {self.strategy}'

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        # Forward pass through the tabular transformer
        if self.missing_tabular:
            missing_mask = x_tab[1]
            x_tab = self.tabular_model(x=x_tab, mask=missing_mask, mask_special=missing_mask)[:, 0, :]
        else:
            print("Forward, x_tab", x_tab.shape)
            x_tab = self.tabular_model(x_tab)[:, 0, :]
        
        # Pass the output through the classification head
        x = self.head(x_tab)
        return x
