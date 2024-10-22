from typing import List, Tuple
from os.path import join
import os
import sys
import torch
from torch import nn
import numpy as np
import random
import monai.transforms as montrans
from monai.transforms import (
    LoadImaged,
    Compose,
    CopyItemsd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    RandFlipd, 
    RandAdjustContrastd,
    RandAffined,
    RandSpatialCropd,
    Resized,
)
from utils.data_transform import *
import torchio as tio

# Disable TF32 on matmul operations.
#torch.backends.cuda.matmul.allow_tf32 = False
# Disable TF32 on cudnn operations.
#torch.backends.cudnn.allow_tf32 = False

# Verify that TF32 is disabled
#print(f"TF32 on matmul operations: {torch.backends.cuda.matmul.allow_tf32}")
#print(f"TF32 on cudnn operations: {torch.backends.cudnn.allow_tf32}")

#TODO: define more transformations if needed with RandomResizedCrop, rotations. 
# In the paper initially what was used was: 
'''
transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
      transforms.Lambda(lambda x: x.float())
    ])
'''

def create_logdir(name: str, resume_training: bool, wandb_logger):
  basepath = join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', name)
  print(basepath)
  if wandb_logger.experiment.name is not None:
    run_name = wandb_logger.experiment.name
    logdir = join(basepath, run_name)
  else:
    logdir = "PATH_TO_REPO/runs/multimodal/test"
  if os.path.exists(logdir) and not resume_training:
    raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
  os.makedirs(logdir,exist_ok=True)
  return logdir

### IMPORTANT NOTE: ###
# LoadImaged with image_only=True is to return the MetaTensors
# the additional metadata dictionary is not returned

def transforms_contrastive_multimodal(img_size: int, augmentation_rate: float, fast=False, device="cuda:0", original_height: int = 182):
    train_transform = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        CenterSpatialCropd(keys=["image"], roi_size=(original_height, original_height, original_height)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        Resized(keys=["gt_image"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False),
        # Apply different transformations to image and image_2:
        RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandAffined(keys=["image"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    if random.random() < augmentation_rate: 
        train_transform.append(RandSpatialCropd(keys=["image_2"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False))
        train_transform.append(RandFlipd(keys=["image_2"], prob=0.5, spatial_axis=(0, 1, 2)))
        train_transform.append(RandAffined(keys=["image_2"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)))
        train_transform.append(ScaleIntensityRangePercentilesd(
            keys=["image_2"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ))
    else :
        train_transform.append(Resized(keys=["image_2"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False))
    return Compose(train_transform)

def transforms_hard_contrastive_multimodal(img_size: int, augmentation_rate: float, fast=False, device="cuda:0", original_height: int = 182):
    train_transform = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        CenterSpatialCropd(keys=["image"], roi_size=(original_height, original_height, original_height)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        Resized(keys=["gt_image"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False),
        # Apply different transformations to image and image_2:
        RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandAffined(keys=["image"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)),
        #RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.5)),  # Adjust contrast randomly
        #RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3)  # Shift intensity (brightness)
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    if random.random() < augmentation_rate: 
        train_transform.append(RandSpatialCropd(keys=["image_2"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False))
        train_transform.append(RandFlipd(keys=["image_2"], prob=0.5, spatial_axis=(0, 1, 2)))
        train_transform.append(RandAffined(keys=["image_2"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)))
        train_transform.append(RandAdjustContrastd(keys=["image_2"], prob=0.3, gamma=(0.5, 1.5)))  # Adjust contrast randomly
        #train_transform.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3))  # Shift intensity (brightness)
        #train_transform.append(RandGaussianNoised(keys=["image_2"], prob=0.5, mean=0.0, std=0.1))
        train_transform.append(ScaleIntensityRangePercentilesd(
            keys=["image_2"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ))
    else :
        train_transform.append(Resized(keys=["image_2"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False))
    return Compose(train_transform)
  
def transforms_last_version(img_size: int, augmentation_rate: float, fast=False, device="cuda:0", original_height: int = 182):
    train_transform = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        CenterSpatialCropd(keys=["image"], roi_size=(original_height, original_height, original_height)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CopyItemsd(keys=["image"], times=1, names=["image_2"], allow_missing_keys=False),
        # Apply different transformations to image_1 and image_2:
        RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandAffined(keys=["image"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    ]
    if random.random() < augmentation_rate: 
        train_transform.append(RandSpatialCropd(keys=["image_2"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False))
        train_transform.append(RandFlipd(keys=["image_2"], prob=0.5, spatial_axis=(0, 1, 2)))
        train_transform.append(RandAffined(keys=["image_2"], prob=0.5, rotate_range=(-90, 90), translate_range=(-8, 8)))
        train_transform.append(ScaleIntensityRangePercentilesd(
            keys=["image_2"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ))
    else :
        train_transform.append(Resized(keys=["image_2"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False))
    return Compose(train_transform)

def transformations_eval(img_size: int, augmentation_rate: float, fast=False, device="cuda:0", original_height: int = 182):
    train_transform = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        CenterSpatialCropd(keys=["image"], roi_size=(original_height, original_height, original_height)),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    if random.random() < augmentation_rate: 
        train_transform.append(RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_center=True, random_size=False))
        train_transform.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=(0, 1, 2)))
        train_transform.append(RandAffined(keys=["image"], prob=0.5, rotate_range=(-8, 8), translate_range=(-8, 8)))
        train_transform.append(ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ))
    else :
        train_transform.append(Resized(keys=["image"], spatial_size=(img_size, img_size, img_size), mode='trilinear', align_corners=False))
    val_transform = [
        LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        CenterSpatialCropd(keys=["image"], roi_size=(original_height, original_height, original_height)),
        ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True,
        ),
        Resized(keys=["image"], spatial_size = (img_size, img_size, img_size), mode='trilinear', align_corners=False)
    ]
    return montrans.Compose(train_transform), montrans.Compose(val_transform)

def grab_arg_from_checkpoint(args: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint)
    load_args = ckpt['hyper_parameters']
  else:
    load_args = args
  return load_args[arg_name]

def chkpt_contains_arg(ckpt_path: str, arg_name: str):
  """
  Checks if a checkpoint contains a given argument.
  """
  ckpt = torch.load(ckpt_path)
  return arg_name in ckpt['hyper_parameters']

def prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams['{}_short'.format(hp)] = hparams[hp]
      hparams[hp] = join(db, hparams[hp])

  return hparams

def re_prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams[hp] = join(db, hparams['{}_short'.format(hp)])

  return hparams