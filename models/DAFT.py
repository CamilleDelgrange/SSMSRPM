'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on TIP codebase https://github.com/siyi-wind/TIP/blob/main/models/Tip_utils/Tip_pretraining.py
* Based on DAFT codebase https://github.com/ai-med/DAFT/tree/master/daft/networks
'''

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks import nets

class DAFT_block(nn.Module):
    def __init__(self, image_dim, tabular_dim, r=7) -> None:
        super(DAFT_block, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        h1 = image_dim + tabular_dim
        h2 = int(h1 / r)
        self.multimodal_projection = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 2 * image_dim)
        )
  
    def forward(self, x_im, x_tab):
        #print(x_im)
        B, C, D, H, W = x_im.shape
        x = self.global_pool(x_im).view(B, -1)
        x = torch.cat([x, x_tab], dim=1)
        attention = self.multimodal_projection(x)
        v_scale, v_shift = torch.split(attention, C, dim=1)
        v_scale = v_scale.view(B, C, 1, 1, 1).expand(-1, -1, D, H, W)
        v_shift = v_shift.view(B, C, 1, 1, 1).expand(-1, -1, D, H, W)
        x = v_scale * x_im + v_shift
        return x

class DAFT(nn.Module):
    """
    Evaluation model for imaging and tabular data.
    """
    def __init__(self, args) -> None:
        super(DAFT, self).__init__()
        self.args = args
        if args.model == 'resnet50':
            #self.imaging_encoder = nets.resnet50(spatial_dims=3, n_input_channels=1, pretrained=False)
            #self.imaging_encoder = nn.Sequential(*list(self.imaging_encoder.children())[:-1])
            self.imaging_encoder = nets.ResNetFeatures('resnet50', pretrained=False, spatial_dims=3, in_channels=1)
            self.imaging_encoder.layer4 = torch.nn.Sequential(*list(self.imaging_encoder.layer4.children())[:-1])
        elif args.model == 'resnet18':
            #print(args.model)
            #self.imaging_encoder = nets.resnet18(spatial_dims=3, n_input_channels=1, pretrained=False)
            #self.imaging_encoder = nn.Sequential(*list(self.imaging_encoder.children())[:-1])
            self.imaging_encoder = nets.ResNetFeatures('resnet18', pretrained=False, spatial_dims=3, in_channels=1)
            self.imaging_encoder.layer4 = torch.nn.Sequential(*list(self.imaging_encoder.layer4.children())[:-1])
        self.tabular_encoder = nn.Identity()
        self.daft = DAFT_block(args.embedding_dim, args.input_size)

        in_ch, out_ch = args.embedding_dim // 4, args.embedding_dim
        self.residual = nn.Sequential(
            nn.Conv3d(out_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch)
        )
        self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        # this depends on the resnet applied
        in_dim = args.embedding_dim
        self.head = nn.Linear(in_dim, args.num_classes)

        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module, init_gain=0.02) -> None:
        """
        Initializes weights according to desired strategy
        """
        if isinstance(m, nn.Linear):
            if self.args.init_strat == 'normal':
                nn.init.normal_(m.weight.data, 0, 0.001)
            elif self.args.init_strat == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif self.args.init_strat == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.args.init_strat == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x_im, x_tab) -> torch.Tensor:
        x_im = self.imaging_encoder(x_im)[-1]
        #print("After imaging encoder", x_im.shape)
        x = self.daft(x_im=x_im, x_tab=x_tab)
        x = self.residual(x)
        x = x + self.shortcut(x_im)
        x = self.act(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.model = 'resnet50'
            self.embedding_dim = 2048
            self.input_size = 90
            self.num_classes = 2
            self.init_strat = 'kaiming'

    args = Args()
    model = DAFT(args)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create dummy input data and move to GPU if available
    img = torch.randn(4, 1, 64, 64, 64).to(device)  # Batch of 4 images, 1 channel, 64x64x64
    tab = torch.randn(4, 90).to(device)  # Batch of 4 tabular inputs, 90 features

    # Run a forward pass
    y = model(img, tab)
    print(y.shape) # correctly displays torch.Size([4, 2])
    