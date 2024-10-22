'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/models/ImagingModel.py
'''

import torch
import torch.nn as nn
from collections import OrderedDict
from monai.networks import nets

class ImagingModel(nn.Module):
  """
  Evaluation model for imaging trained with ResNet encoder.
  """
  def __init__(self, args) -> None:
    super(ImagingModel, self).__init__()

    if args.checkpoint:
      if args.checkpoint_imaging or args.checkpoint_multimodal:
        # Load weights
        checkpoint = torch.load(args.checkpoint)
        original_args = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']
        #checkpoint = self.load_model_checkpoint(args.checkpoint)

        if 'encoder_imaging.0.weight' in state_dict:
          self.bolt_encoder = False
          self.encoder_name = 'encoder_imaging.'
          self.encoder = self.create_imaging_model(original_args['model'])
        elif 'encoder.' in state_dict:
          self.bolt_encoder = False
          self.encoder_name = 'encoder.'
          self.encoder = self.create_imaging_model(original_args['model'])
        else:
          encoder_name_dict = {'clip' : 'encoder_imaging.', 'barlowtwins': 'network.encoder.'}
          self.bolt_encoder = True
          self.encoder = self.create_imaging_model(original_args['model'])
          self.encoder_name = encoder_name_dict[original_args['loss']]
        
        # Remove prefix and fc layers
        state_dict_encoder = {}
        for k in list(state_dict.keys()):
          if k.startswith(self.encoder_name) and not 'projection_head' in k and not 'prototypes' in k:
            state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]
        
        #self.compare_state_dicts(args.model, args.checkpoint)
        log = self.encoder.load_state_dict(state_dict_encoder, strict=True)
        assert len(log.missing_keys) == 0

        # Freeze if needed
        if args.finetune_strategy == 'frozen':
          print("Frozen finetuning")
          for _, param in self.encoder.named_parameters():
            param.requires_grad = False
          parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
          assert len(parameters)==0
        # Partial unfreezing strategy
        if args.finetune_strategy == 'partial_unfreeze':
          print("Partial Unfreezing")
          layers_to_unfreeze = [f'{self.encoder_name}.7.']  # Unfreezing up to layer 7
          for name, param in self.encoder.named_parameters():
            if any(layer_name in name for layer_name in layers_to_unfreeze):
              param.requires_grad = True
              print(f"Unfreezing layer: {name}")
      elif args.checkpoint_tabular:
        self.bolt_encoder = True
        self.pooled_dim = 2048 if args.model=='resnet50' else 512
        self.encoder = self.create_imaging_model(args.model)
    else:
      self.bolt_encoder = True
      self.pooled_dim = 2048 if args.model=='resnet50' else 512
      self.encoder = self.create_imaging_model(args.model)
    
    # Add a global average pooling layer
    #self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
    #self.dropout = nn.Dropout(p=args['dropout_rate']) # Add more regularization
    self.classifier = nn.Linear(self.pooled_dim, args.num_classes)

  def create_imaging_model(self, model_name):
    if model_name == 'resnet18':
        encoder = nets.resnet18(spatial_dims=3, n_input_channels=1, pretrained=False)
        self.pooled_dim = 512
    elif model_name == 'resnet50':
        encoder = nets.resnet50(spatial_dims=3, n_input_channels=1, pretrained=False)
        self.pooled_dim = 2048
    else:
        raise Exception('Invalid architecture. Please select either resnet18 or resnet50.')
    encoder = nn.Sequential(*list(encoder.children())[:-1])
    return encoder

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = x.view(x.size(0), -1) 
    x = self.classifier(x)
    return x