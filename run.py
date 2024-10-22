import warnings

# Suppress specific warning from monai.utils.tf32
warnings.filterwarnings(
    "ignore",
    message="torch.backends.cuda.matmul.allow_tf32 = True by default.\n  This value defaults to True when PyTorch version in [1.7, 1.11] and may affect precision.\n  See https://docs.monai.io/en/latest/precision_accelerating.html#precision-and-accelerating",
    category=UserWarning,
    module="monai.utils.tf32"
)

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys
import time
from datetime import datetime
import random

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
print("Cuda version", torch.version.cuda)
print("Cudnn backend version", torch.backends.cudnn.version())
print("Cuda nccl version", torch.cuda.nccl.version())
torch.cuda.memory_summary(device="cuda:0", abbreviated=False)

import torch.multiprocessing
# Set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy('file_descriptor')

#Enable cuDNN auto-tuner to find the best/most efficient algo for convs:
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths


#@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  #print("Arguments received in the 'run' function:")
  #print(args)
  now = datetime.now()
  start = time.time()
  pl.seed_everything(args.seed)
  args = prepend_paths(args)
  time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once

  if args.resume_training:
    if args.wandb_id:
      wandb_id = args.wandb_id
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    args = OmegaConf.create(args)
    args.checkpoint = checkpoint
    args.resume_training = True
    if not 'wandb_id' in args or not args.wandb_id:
      args.wandb_id = wandb_id
    # Run prepend again in case we move to another server and need to redo the paths
    args = re_prepend_paths(args)
  
  if args.generate_embeddings:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'dataset')
    generate_embeddings(args)
    return args
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  exp_name = f'{args.exp_name}_{args.target}_{now.strftime("%m%d_%H%M")}'
  if args.use_wandb:
    if args.resume_training and args.wandb_id:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline, id=args.wandb_id, resume='must')
    else:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline)
  else:
    wandb_logger = WandbLogger(name=exp_name, project='Test', entity='cdelgrange', save_dir=base_dir, offline=args.offline)
  args.wandb_id = wandb_logger.version
  
  # Log the original run name as a hyperparameter
  wandb_logger.experiment.config.update({"original_run_name": wandb_logger.experiment.name})

  if args.checkpoint and not args.resume_training:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'datatype')

  #print('Comment: ', args.comment)
  print(f'Pretrain LR: {args.lr}, Decay: {args.weight_decay}')
  print(f'Finetune LR: {args.lr_eval}, Decay: {args.weight_decay_eval}')
      
  if args.pretrain:
    print('=================================================================================\n')
    print('Start pretraining\n')  
    print('=================================================================================')
    pretrain(args, wandb_logger)
    args.checkpoint = os.path.join(base_dir, 'runs', args.datatype, wandb_logger.experiment.name, f'checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt')
  
  if args.test:
    test(args, wandb_logger)
  elif args.evaluate:
    print('=================================================================================\n')
    print('Start Finetuning')  
    print('=================================================================================\n')
    evaluate(args, wandb_logger)


  wandb.finish()
  del wandb_logger
  end = time.time()
  time_elapsed = end-start
  print('Total running time: {:.0f}h {:.0f}m'.
      format(time_elapsed // 3600, (time_elapsed % 3600)//60))

@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config_pretraining', version_base=None)
def control(args: DictConfig):
  #print("Arguments received in the 'control' function:")
  #print(args)
  run(args)

if __name__ == "__main__":
  control()