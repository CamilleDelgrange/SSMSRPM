'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/trainers/pretrain.py
'''

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #garbage_collection_threshold:0.6,
import sys
import psutil
import torch
import time
import monai

#from torchsummary import summary
#from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.strategies import DDPStrategy

from utils.utils import transforms_hard_contrastive_multimodal, transforms_contrastive_multimodal, transforms_last_version, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset
from datasets.ContrastiveImagingAndTabularDatasetCached import ContrastiveImagingAndTabularDatasetCached
from models.MultimodalSimCLR import MultimodalSimCLR
from models.SimCLR import SimCLR
from models.BarlowTwins3D import Barlow_Twins_Module
from models.SCARF import SCARF
from models.Model2Loss import Model2Loss
from monai.data import (
    DataLoader,
    Dataset,
    set_track_meta
)

# Disable TF32 on matmul operations.
#torch.backends.cuda.matmul.allow_tf32 = False
# Disable TF32 on cudnn operations.
#torch.backends.cudnn.allow_tf32 = False

# Verify that TF32 is disabled
#print(f"TF32 on matmul operations: {torch.backends.cuda.matmul.allow_tf32}")
#print(f"TF32 on cudnn operations: {torch.backends.cudnn.allow_tf32}")

def get_object_size(obj):
    return sys.getsizeof(obj)

def check_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB, VMS: {mem_info.vms / (1024 ** 2):.2f} MB")

def get_available_vram():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        vram_available = torch.cuda.get_device_properties(device).total_memory
        vram_allocated = torch.cuda.memory_allocated(device)
        vram_free = vram_available - vram_allocated
        return device, vram_available, vram_allocated, vram_free
    else:
        return "CUDA is not available"    

def load_datasets(hparams, device=None, reload=False):
  if hparams.datatype == 'multimodal':
    set_track_meta(False)
    torch.cuda.empty_cache()
    train_transforms = transforms_hard_contrastive_multimodal(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    
    train_ds = ContrastiveImagingAndTabularDatasetCached(
      hparams.data_train_imaging, hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot, hparams.labels_train)
    #print("Len train dataset", len(train_ds))
    #print(f"Total Train Dataset Size: {total_dataset_size_train} bytes")
    #print("memory check before train init")
    #check_memory_usage()
    train_dataset = Dataset(
        data=train_ds,
        transform=train_transforms, 
    )
  
    #print("memory check after train init")
    #check_memory_usage()
    
    val_ds = ContrastiveImagingAndTabularDatasetCached(
      hparams.data_val_imaging, hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot, hparams.labels_val)
    val_dataset = Dataset(
        data=val_ds, 
        transform=train_transforms
    )
    
    hparams.input_size = train_ds.get_input_size()
  elif hparams.datatype == 'imaging':
    train_transforms = transforms_last_version(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    # Load the tensors
    train_image_paths = torch.load(hparams.data_train_imaging)
    train_labels = torch.load(hparams.labels_train)
    val_image_paths = torch.load(hparams.data_val_imaging)
    val_labels = torch.load(hparams.labels_val)
    # Create data dictionaries
    train_data = [{"image": str(path), "label": int(label)} for path, label in zip(train_image_paths, train_labels)]
    val_data = [{"image": str(path), "label": int(label)} for path, label in zip(val_image_paths, val_labels)]
    train_dataset = Dataset(data=train_data, transform=train_transforms) 
    val_dataset = Dataset(data=val_data, transform=train_transforms) 
  elif hparams.datatype == 'tabular':
    train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset


def select_model(hparams, train_dataset, device=None): 
  if hparams.datatype == 'multimodal':
    if hparams.strategy == 'itm':
      # ITM
      model = Model2Loss(hparams)
      print('Using Model2Loss')
    else:
      # MMCL
      model = MultimodalSimCLR(hparams) 
  elif hparams.datatype == 'imaging':
    if hparams.loss.lower() == 'barlowtwins':
      def filter_hparams(hparams):
        allowed_keys = {"in_channels", "n_blocks", "dropout_rate", "bn_momentum", "n_basefilters", 
                    "resnet_version", "encoder_out_dim", "z_dim", "lambda_coeff", "batch_size", 
                    "encoder_num_layers", "pretrained_model", "warmup_epochs", "anneal_max_epochs", "lr", 
                    "weight_decay"}
        filtered_hparams = {k: v for k, v in hparams.items() if k in allowed_keys}
        return filtered_hparams
      # When selecting the model, filter the hyperparameters
      filtered_hparams = filter_hparams(hparams)
      model = Barlow_Twins_Module(**filtered_hparams)
    else:
      model = SimCLR(hparams)
  elif hparams.datatype == 'tabular':
    model = SCARF(hparams)
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model

def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  
  if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
  else:
      raise RuntimeError("this tutorial is intended for GPU, but no CUDA device is available")
  
  device, vram_available, vram_allocated, vram_free = get_available_vram()
  print("Device", device, "Vram available", vram_available, "Vram allocated", vram_allocated, "Vram free", vram_free)
  pl.seed_everything(hparams.seed)

  print("Check memory before loading train & val")
  check_memory_usage()  
  
  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size, shuffle=True, pin_memory = False, persistent_workers=True) 
  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size, shuffle=False, pin_memory = False, persistent_workers=True) 
  model = select_model(hparams, train_dataset) 

  print("Training dataset size:", len(train_dataset))
  print("Validation dataset size:", len(val_dataset))
  print("Number of batches in training DataLoader:", len(train_loader))
  print("Number of batches in validation DataLoader:", len(val_loader))
  print("Batch size", hparams.batch_size)
  
  print("Check memory after loading train & val")
  check_memory_usage()  
  
  # Create logdir based on WandB run name
  print(wandb_logger)
  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  print(logdir)

  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
  callbacks.append(ModelCheckpoint(
    dirpath=logdir,
    filename=f'checkpoint_last_epoch_{{epoch:02d}}_{{{hparams.datatype}.val.loss:.2f}}',
    monitor=f'{hparams.datatype}.val.loss',  
    save_top_k=3, 
    mode='min', 
    auto_insert_metric_name=False
  ))
  # Checkpoint to save the last model at the end of the last epoch
  callbacks.append(ModelCheckpoint(
      dirpath=logdir,
      filename='checkpoint_last_epoch_{epoch:02d}',
      save_last=True,  # This will ensure the last model is saved
      auto_insert_metric_name=False
  ))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  # Define your profiler and put it in Trainer if needed:
  profiler = AdvancedProfiler(dirpath="/opt/notebooks/MMCL/profiling/", filename="profile_output_wo_dicts")
  trainer = Trainer.from_argparse_args(hparams, accumulate_grad_batches = 8, accelerator='gpu', devices = 1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, log_every_n_steps=hparams.log_every_n_steps, check_val_every_n_epoch=hparams.check_val_every_n_epoch, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar, precision=16)
  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    device, vram_available, vram_allocated, vram_free = get_available_vram()
    print("Device", device, "Vram available", vram_available, "Vram allocated", vram_allocated, "Vram free", vram_free)
    trainer.fit(model, train_loader, val_loader)