from typing import List, Tuple, Dict
import numpy as np
import torch

from utils.ntx_ent_loss_custom import NTXentLoss
from utils.clip_loss import CLIPLoss

from models.pretraining import Pretraining


class MultimodalSimCLR(Pretraining):
  """
  Lightning module for multimodal SimCLR.
  """
  def __init__(self, hparams):
    super().__init__(hparams)
    self.should_skip_lr_scheduler_step = False

    # Imaging
    self.initialize_imaging_encoder_and_projector()

    if self.hparams.imaging_pretrain_checkpoint:
      self.load_pretrained_imaging_weights()

    # Tabular
    self.initialize_tabular_encoder_and_projector()

    # Multimodal
    nclasses = hparams.batch_size
    self.criterion_val = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0, hard_neg=False)
    if self.hparams.loss.lower() == 'clip':
      self.criterion_train = self.criterion_val
    elif self.hparams.loss.lower() == 'ntxent':
      self.criterion_train = NTXentLoss(self.hparams.temperature)
      self.criterion_val = self.criterion_train
      nclasses = hparams.batch_size*2-1
    else:
      raise ValueError('The only implemented losses currently are CLIP and NTXent')

    self.initialize_classifier_and_metrics(nclasses, nclasses)

    #print(f'Tabular model, multimodal: {self.encoder_tabular}\n{self.projector_tabular}')
    #print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projector_imaging}')   

  def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Trains contrastive model
    """
    im_view, tab_views, y = batch['image_2'], batch['tabular_views'], batch['label']
   
    z0, image_embeddings, image_embeddings_flatt = self.forward_imaging(im_view) # returns latent projected + embeddings for downstream tasks
    z1, tabular_embeddings, tabular_embeddings_flatt = self.forward_tabular(tab_views[1]) # returns latent projected but no embeddings because we won't use tabular data for downstream tasks
    loss, logits, labels = self.criterion_train(z0, z1, y)
  
    self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)

    if len(im_view)==self.hparams.batch_size:
      self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='multimodal')

    return {'loss': loss, 'embeddings': image_embeddings_flatt, 'labels': y}

  def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Validate contrastive model
    """

    im_view, original_image, tab_views, y = batch['image_2'], batch['gt_image'], batch['tabular_views'], batch['label']

    z0, image_embeddings, image_embeddings_flatt = self.forward_imaging(original_image)
    z1, tabular_embeddings, tabular_embeddings_flatt = self.forward_tabular(tab_views[0])
    loss, logits, labels = self.criterion_val(z0, z1, y)
    
    self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
    if len(im_view)==self.hparams.batch_size:
      self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='multimodal')

    return {'embeddings': image_embeddings_flatt, 'labels': y}

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model. 
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projector_imaging.parameters()},
        {'params': self.encoder_tabular.parameters()},
        {'params': self.projector_tabular.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    scheduler = self.initialize_scheduler(optimizer)

    return {
          'optimizer': optimizer,
          'lr_scheduler': {
              'scheduler': scheduler
          }
      }
