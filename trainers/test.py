'''
* Licensed under the Apache License, Version 2.
* By Camille Delgrange, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/trainers/test.py
'''


from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImagingAndTabularDatasetEval import ImagingAndTabularDatasetEval
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
import torch 
from utils.utils import transformations_eval

from monai.data import Dataset 

def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
  else:
      raise RuntimeError("This is intended for GPU, but no CUDA device is available")
  pl.seed_everything(hparams.seed)
  
  if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
    # Load the tensors
    test_image_paths = torch.load(hparams.data_test_eval_imaging)
    test_labels = torch.load(hparams.labels_test_eval_imaging)
    _, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    # Create data dictionaries
    test_data = [{"image": str(path), "label": int(label)} for path, label in zip(test_image_paths, test_labels)]
    test_dataset = Dataset(data=test_data, transform=val_transforms)
    '''
    test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task)
    hparams.transform_test = test_dataset.transform_val.__repr__()
    '''
  elif hparams.datatype == 'tabular':
    test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.datatype == 'imaging_and_tabular':
    _, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    test_ds = ImagingAndTabularDatasetEval(
      hparams.data_test_eval_imaging, hparams.data_test_eval_tabular, 0.0, hparams.field_lengths_tabular, hparams.one_hot, hparams.labels_test_eval_imaging)
    test_dataset = Dataset(
        data=test_ds,
        transform=val_transforms, 
    )
    hparams.input_size = test_ds.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  drop = ((len(test_dataset)%hparams.batch_size)==1)
  print(drop)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  hparams.dataset_length = len(test_loader)

  model = Evaluator(hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)