# This file is part of From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data (Triplet Training).
#
# Triplet Training is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Triplet Training is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Triplet Training. If not, see <https://www.gnu.org/licenses/>.

import logging
from typing import Any
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from utils.BarlowUtils import BarlowTwinsLoss, ProjectionHead
from models.optimizer import LARS, CosineWarmupScheduler
from models.resnet import BaseResNet, SingleResNet

#LOG = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def load_pretrain_model(pretrained_model):
    pretrained_dict = torch.load(pretrained_model)["state_dict"]
    new_pretrained_dict = dict()
    for key, value in pretrained_dict.items():
        new_pretrained_dict[key[4:]] = value
    return new_pretrained_dict

class Barlow_Twins_Module(pl.LightningModule):
    """
    Lightning module for 3D implementation of BarlowTwins
    """
    def __init__(
        self,
        pretrained_model: str,
        lr: float = 0.5,
        weight_decay: float = 1.5e-6,
        batch_size: int = 6,
        encoder_out_dim: int = 512,
        z_dim: int = 2048,
        lambda_coeff: float = 5e-3,
        warmup_epochs: int = 10,
        anneal_max_epochs: int = 100,
        in_channels: int = 1, 
        n_blocks: int = 6, 
        bn_momentum: float = 0.05,
        n_basefilters: int = 16, 
        dropout_rate: float = 0.1,
        resnet_version: str = 'base', 
        encoder_num_layers: int = 2, 
        **kwargs: Any,

    )-> None:
        super().__init__()
        self.save_hyperparameters()
        # Initialize ResNet encoder based on config
        self.encoder = SingleResNet(
            in_channels=in_channels,
            n_blocks=n_blocks,
            bn_momentum=bn_momentum,
            n_basefilters=n_basefilters,
            dropout_rate=dropout_rate,
            resnet_version=resnet_version
        ).to(device)

        self.encoder.apply(init_weights)

        if pretrained_model != None:
            new_pretrained_dict = load_pretrain_model(pretrained_model)
            missing_keys, unexpected_keys = self.encoder.load_state_dict(new_pretrained_dict, strict = False)              
            print('======load pretrained model successfully========')
            print("missing keys:", msg.missing_keys)
            print(f"Unexpected Keys: {unexpected_keys}")

        self.projection_head = ProjectionHead(
                                num_layer = encoder_num_layers,
                                input_dim=encoder_out_dim, 
                                hidden_dim=encoder_out_dim * 2, 
                                output_dim=z_dim,
                                last_bn=True)

        # loss function 
        self.barlow_loss = BarlowTwinsLoss(batch_size=batch_size, 
                                            lambda_coeff=lambda_coeff, 
                                            z_dim=z_dim)


    def forward(self, x: torch.Tensor):
        return self.encoder(x, get_features = True)

    def shared_step(self, batch):
        x1, x2, y = batch['image'], batch['image_2'], batch['label']
        print("Batch out", x1.shape, x2.shape)
        print("Forward", self.forward(x1).shape, self.forward(x1).shape)
        z1 = self.projection_head(self.forward(x1))
        z2 = self.projection_head(self.forward(x2))
        print("After projection", z1.shape, z2.shape)

        return self.barlow_loss(z1, z2)

    def training_step(self, batch: Any, batch_idx: int):       
        loss = self.shared_step(batch)
        self.log(f"imaging.train.loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):       
        loss = self.shared_step(batch)
        self.log(f"imaging.val.loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}
    '''
    def test_step(self, batch: Any, batch_idx: int):
        loss = self.shared_step(batch)
        self.log("imaging.test.loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}
    '''
    def exclude_bias_and_norm(self, p):
        """
        This function returns True if the parameter should be excluded from weight decay,
        i.e., if it is a bias term or a norm term (like batch normalization or layer normalization).
        """
        return p.ndim == 1

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = LARS(
            self.parameters(),
            lr=0,  # Initialize with a LR of 0
            weight_decay=self.hparams.weight_decay,
            weight_decay_filter=self.exclude_bias_and_norm,
            lars_adaptation_filter=self.exclude_bias_and_norm
        )

        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            batch_size=self.hparams.batch_size,
            warmup_steps=self.hparams.warmup_epochs,
            max_steps=self.hparams.anneal_max_epochs,
            lr=self.hparams.lr,
            end_lr=0.0005,
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}