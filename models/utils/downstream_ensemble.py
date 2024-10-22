'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''

import sys
import os 
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# Append the path to the root of the repository to sys.path
REPO_PATH = os.path.abspath(os.path.join(CURRENT_PATH, '..'))
sys.path.append(REPO_PATH)

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig, open_dict, OmegaConf
from monai.networks import nets

from models.utils.Transformer import MultimodalTransformerEncoder
from utils.pieces import DotDict
#from models.utils.VisionTransformer_imagenet import create_vit
from models.TabularEncoderITM import TabularEncoder

class BackboneEnsemble(nn.Module):
    """
    Evaluation model.
    """
    def __init__(self, args) -> None:
        super(BackboneEnsemble, self).__init__()
        self.missing_tabular = args.missing_tabular
        print(f'Current missing tabular for BackboneEnsemble: {self.missing_tabular}')
        if args.checkpoint:
            print(f'Checkpoint name: {args.checkpoint}')
            # Load weights
            checkpoint = torch.load(args.checkpoint)
            original_args = OmegaConf.create(checkpoint['hyper_parameters'])
            original_args.field_lengths_tabular = args.field_lengths_tabular
            original_args.checkpoint = args.checkpoint
            if 'checkpoint_multimodal' not in original_args:
                with open_dict(original_args):
                    original_args.checkpoint_multimodal = args.checkpoint_multimodal
            if 'checkpoint_imaging' not in original_args:
                with open_dict(original_args):
                    original_args.checkpoint_imaging = args.checkpoint_imaging
            if 'checkpoint_tabular' not in original_args:
                with open_dict(original_args):
                    original_args.checkpoint_tabular = args.checkpoint_tabular
            if 'algorithm_name' not in original_args:
                with open_dict(original_args):
                    original_args.algorithm_name = args.algorithm_name
            state_dict = checkpoint['state_dict']
            self.hidden_dim = original_args.multimodal_embedding_dim
            self.pooled_dim = original_args.embedding_dim
            self.tab_dim = original_args.tabular_embedding_dim

            # load image encoder
            if 'encoder_imaging.0.weight' in state_dict:
                self.encoder_name_imaging = 'encoder_imaging.'
            else:
                encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
                self.encoder_name_imaging = encoder_name_dict[original_args.loss]

            #if original_args.model.startswith('vit'):
                #self.encoder_imaging = create_vit(original_args)
                #self.encoder_imaging_type = 'vit'
            if original_args.model.startswith('resnet'):
                model = nets.resnet50(spatial_dims=3, n_input_channels=1)
                self.encoder_imaging = nn.Sequential(*list(model.children())[:-2]) # remove classifier layer + adaptive pool avg
                self.encoder_imaging_type = 'resnet'
            
            # load tabular encoder
            self.create_tabular_model(original_args)
            self.encoder_name_tabular = 'encoder_tabular.'
            #assert len(self.cat_lengths_tabular) == original_args.num_cat
            #assert len(self.con_lengths_tabular) == original_args.num_con
            # load multimodal encoder
            self.create_multimodal_model(original_args)
            self.encoder_name_multimodal = 'encoder_multimodal.'

            for module, module_name in zip([self.encoder_imaging, self.encoder_tabular, self.encoder_multimodal], 
                                            [self.encoder_name_imaging, self.encoder_name_tabular, self.encoder_name_multimodal]):
                self.load_weights(module, module_name, state_dict)
                if args.finetune_strategy == 'frozen':
                    for _, param in module.named_parameters():
                        param.requires_grad = False
                    parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
                    assert len(parameters)==0
                    print(f'Freeze {module_name}')
                elif args.finetune_strategy == 'trainable':
                    print(f'Full finetune {module_name}')
                else:
                    assert False, f'Unknown finetune strategy {args.finetune_strategy}'

        else:
            self.create_imaging_model(args)
            self.create_tabular_model(args)
            self.create_multimodal_model(args)
            self.hidden_dim = args.multimodal_embedding_dim
            self.pooled_dim = args.embedding_dim
            self.tab_dim = args.tabular_embedding_dim

        self.classifier_multimodal = nn.Linear(self.hidden_dim, args.num_classes)
        self.classifier_imaging = nn.Linear(self.pooled_dim, args.num_classes)
        self.classifier_tabular = nn.Linear(self.tab_dim, args.num_classes)

    def create_imaging_model(self, args):
        #if args.model.startswith('vit'):
            #self.encoder_imaging = create_vit(args)
            #self.encoder_imaging_type = 'vit'
        if args.model.startswith('resnet'):
            model = nets.resnet50(spatial_dims=3, n_input_channels=1)
            self.encoder_imaging = nn.Sequential(*list(model.children())[:-2]) # remove classifier layer + adaptive pool avg
            self.encoder_imaging_type = 'resnet'

    def create_multimodal_model(self, args):
        self.encoder_multimodal = MultimodalTransformerEncoder(args)

    def create_tabular_model(self, args):
        self.encoder_tabular = TabularEncoder(args) #don't forget to put tabular_embedding_dim in encoder!!

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    def forward(self, x_im: torch.Tensor, x_tab: torch.Tensor, visualize=False) -> torch.Tensor:
        x_i, x_t = x_im, x_tab
        x_i = self.encoder_imaging(x_i)  # (B,C,H,W) # they were doing [-1] because they return all_feature_maps=True from the 
        # torchvision_ssl_encoder!!
        # To have seq length of 1 & avoid too many arguments:
        x_i = F.adaptive_avg_pool3d(x_i, (1, 1, 1)).flatten(1)
        # missing mask
        if self.missing_tabular:
            missing_mask = x[2]
            x_t = self.encoder_tabular(x=x_t, mask=missing_mask, mask_special=missing_mask)
        else:
            #print(x_t.shape)
            x_t = self.encoder_tabular(x_t)
            #print(x_t.shape)
            x_t = x_t.flatten(start_dim=1)
            #print(x_t.shape)   # (B,N_t,C) with N_t = 1 in this case.
        if visualize==False:
            x_m = self.encoder_multimodal(x=x_t, image_features=x_i)
        else:
            x_m, attn = self.encoder_multimodal(x=x_t, image_features=x_i, visualize=visualize)

        if self.encoder_imaging_type == 'resnet':
            out_i = self.classifier_imaging(x_i)
        elif self.encoder_imaging_type == 'vit':
            out_i = self.classifier_imaging(x_i[:,0,:])
        #out_t = self.classifier_tabular(x_t[:,0,:])
        out_t = self.classifier_tabular(x_t)
        out_m = self.classifier_multimodal(x_m[:,0,:])
        x = (out_i+out_t+out_m)/3.0

        if visualize == False:
            return x
        else:
            return x, attn


if __name__ == "__main__":
  args = DotDict({'model': 'resnet50', 'checkpoint': None, 
                  'num_cat': 26, 'num_con': 49, 'num_classes': 2, 'finetune_strategy': 'frozen',
                  'field_lengths_tabular': '/cluster/work/grlab/projects/tmp_cdelgrange/FINAL_SPLITS/field_lengths_tabular.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 2,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'embedding_dim': 2048, 'input_size': 17, 'encoder_num_layers': 2, 'dropout_rate': 0.0,
                    'multimodal_embedding_dim': 256, 'multimodal_transformer_num_layers': 2})
  model = BackboneEnsemble(args)
  x_i = torch.randn(2, 1, 128, 128, 128)
  x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
  #mask = torch.zeros_like(x_t, dtype=torch.bool)
  x = model(x_im=x_i, x_tab=x_t)
  print(x, x.shape)