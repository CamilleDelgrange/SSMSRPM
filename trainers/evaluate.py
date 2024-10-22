import os 
import torch
import time
import numpy as np
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import Sampler
import psutil
from datasets.ImagingAndTabularDatasetEval import ImagingAndTabularDatasetEval
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from utils.utils import transformations_eval, grab_arg_from_checkpoint, create_logdir
from monai.data import Dataset 

class StratifiedBatchSampler(Sampler):
  def __init__(self, labels, batch_size):
    print("Refined version for length computation")
    self.labels = np.array(labels)
    self.batch_size = batch_size
    self.labels_set = np.unique(labels)
    self.label_to_indices = {label: np.where(labels == label)[0].tolist() for label in self.labels_set}
    self.batches = self._create_batches()

  def _create_batches(self):
    # List to hold batches
    batches = []
    
    # Ensure the loop runs until all indices are exhausted
    while any(len(indices) > 0 for indices in self.label_to_indices.values()):
        batch = []
        for label in self.labels_set:
          label_indices = self.label_to_indices[label]
          batch_size_per_label = min(len(label_indices), self.batch_size // len(self.labels_set))
          selected_indices = label_indices[:batch_size_per_label]
          batch.extend(selected_indices)
          # Update the indices list by removing selected indices
          self.label_to_indices[label] = label_indices[batch_size_per_label:]
        np.random.shuffle(batch)  # Shuffle the combined batch
        if len(batch) == self.batch_size:
          batches.append(batch)
    
    return batches

  def __iter__(self):
    # Shuffle batches to ensure each epoch gets different order
    np.random.shuffle(self.batches)
    for batch in self.batches:
      yield batch

  def __len__(self):
    # Return the count of full batches
    return len(self.batches)

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

def load_datasets(hparams, device):
  device, vram_available, vram_allocated, vram_free = get_available_vram()
  print(device, vram_available, vram_allocated, vram_free)
  if hparams.eval_datatype == 'imaging' or hparams.eval_datatype == 'multimodal':
    # Load the tensors:
    train_image_paths = torch.load(hparams.data_train_eval_imaging)
    train_labels = torch.load(hparams.labels_train_eval_imaging)
    val_image_paths = torch.load(hparams.data_val_eval_imaging)
    val_labels = torch.load(hparams.labels_val_eval_imaging)
    
    train_transforms, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    
    # Create data dictionaries
    train_data = [{"image": str(path), "label": int(label)} for path, label in zip(train_image_paths, train_labels)]
    val_data = [{"image": str(path), "label": int(label)} for path, label in zip(val_image_paths, val_labels)]
    print("Len train images", len(train_image_paths), "Len train labels", len(train_labels))
    print("Len val images", len(val_image_paths), "Len val labels", len(val_labels))
    train_dataset = Dataset(data=train_data, transform=val_transforms)
    val_dataset = Dataset(data=val_data, transform=val_transforms)
  elif hparams.eval_datatype == 'tabular':
    train_dataset = TabularDataset(hparams.data_train_eval_tabular, hparams.labels_train_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    val_dataset = TabularDataset(hparams.data_val_eval_tabular, hparams.labels_val_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.eval_datatype == 'imaging_and_tabular':
    train_transforms, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
    train_ds = ImagingAndTabularDatasetEval(
      hparams.data_train_eval_imaging, hparams.data_train_eval_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.eval_one_hot, hparams.labels_train_eval_imaging)
    train_dataset = Dataset(
        data=train_ds,
        transform=train_transforms, 
    )
    val_ds = ImagingAndTabularDatasetEval(
      hparams.data_val_eval_imaging, hparams.data_val_eval_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.eval_one_hot, hparams.labels_val_eval_imaging)
    val_dataset = Dataset(
        data=val_ds,
        transform=val_transforms, 
    )
    hparams.input_size = train_ds.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular, multimodal or imaging_and_tabular')
  return train_dataset, val_dataset


def evaluate(hparams, wandb_logger):
  """
  Evaluates trained contrastive models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
  else:
      raise RuntimeError("This is intended for GPU, but no CUDA device is available")
  pl.seed_everything(hparams.seed)
  
  train_dataset, val_dataset = load_datasets(hparams, device=device)
  
  drop = ((len(train_dataset)%hparams.batch_size)==1)
  print("Drop if drop last for stability", drop)

  if hparams.stratified_sampler: 
    print('Using stratified sampler')
    train_labels = torch.load(hparams.labels_train_eval_imaging)
    train_sampler = StratifiedBatchSampler(train_labels, batch_size=hparams.batch_size)

  # StratifiedBatchSampler -> mutually exclusive with drop_last, shuffle and batch_size
  train_loader = DataLoader(
    train_dataset, 
    num_workers=hparams.num_workers, batch_sampler=train_sampler,
    pin_memory=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,
    pin_memory=True, shuffle=False, persistent_workers=True)

  def get_label_counts(loader, num_classes):
    overall_counts = np.zeros(num_classes, dtype=int)
    for i, batch in enumerate(loader):
        labels = batch['label'].cpu().numpy()
        batch_counts = np.bincount(labels, minlength=num_classes)
        overall_counts += batch_counts
        print(f"Batch {i + 1} label distribution: {batch_counts}")
    return overall_counts

  logdir = create_logdir('eval', hparams.resume_training, wandb_logger)

  model = Evaluator(hparams)
  mode = 'max'

  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'eval.val.{hparams.eval_metric}', mode=mode, filename=f'checkpoint_best_{hparams.eval_metric}', dirpath=logdir))
  callbacks.append(EarlyStopping(monitor=f'eval.val.{hparams.eval_metric}', min_delta=0.0002, patience=int(10*(1/hparams.val_check_interval)), verbose=False, mode=mode))
  callbacks.append(EarlyStopping(
    monitor=f'eval.val.{hparams.eval_metric}',
    min_delta=0.0001,  
    patience=15, 
    mode=mode
  ))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))
  trainer = Trainer.from_argparse_args(hparams, precision=16, accelerator="gpu", devices=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, log_every_n_steps=hparams.log_every_n_steps, val_check_interval=hparams.val_check_interval, check_val_every_n_epoch=hparams.check_val_every_n_epoch ,limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches)
  trainer.fit(model, train_loader, val_loader)

  # Ensure best_val_score is a standard Python float
  best_val_score = model.best_val_score.item() if isinstance(model.best_val_score, torch.Tensor) else model.best_val_score
  wandb_logger.log_metrics({f'best.val.{hparams.eval_metric}': best_val_score})
  
  if hparams.youden_index_eval:
    hparams.youden_index = True
    hparams.fig_dir = logdir 
    model.freeze()  
    trainer.validate(model, val_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt'))
  
  if hparams.test_and_eval:
    if hparams.eval_datatype == 'imaging' or hparams.eval_datatype == 'multimodal':
      # Load the tensors
      test_image_paths = torch.load(hparams.data_test_eval_imaging)
      test_labels = torch.load(hparams.labels_test_eval_imaging)
      print("----TESTING PHASE----")
      print("Len TEST images", len(test_image_paths), "Len TEST labels", len(test_labels))
      _, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
      # Create data dictionaries
      test_data = [{"image": str(path), "label": int(label)} for path, label in zip(test_image_paths, test_labels)]
      test_dataset = Dataset(data=test_data, transform=val_transforms)

    elif hparams.eval_datatype == 'tabular':
      test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
      hparams.input_size = test_dataset.get_input_size()
    elif hparams.eval_datatype == 'imaging_and_tabular':
      _, val_transforms = transformations_eval(hparams.img_size, hparams.augmentation_rate, False, device, hparams.original_height)
      test_ds = ImagingAndTabularDatasetEval(
        hparams.data_test_eval_imaging, hparams.data_test_eval_tabular, 0.0, hparams.field_lengths_tabular, hparams.eval_one_hot, hparams.labels_test_eval_imaging)
      test_dataset = Dataset(
          data=test_ds,
          transform=val_transforms, 
      )
      hparams.input_size = test_ds.get_input_size()
    else:
      raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    # Verify the size of the test dataset
    print(f"Size of test dataset: {len(test_dataset)}")

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=hparams.batch_size,
      pin_memory=True, shuffle=False, drop_last = drop, persistent_workers=False)

    # Verify the number of batches
    print(f"Number of batches in test DataLoader: {len(test_loader)}")
    
    model.freeze()
    hparams.youden_index = True
    hparams.fig_dir = logdir 
    trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
    trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt'))