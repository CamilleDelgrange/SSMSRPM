# SSMSRPM
GitHub repository for the code of the project and paper "A Self-Supervised Model for Multi-modal Stroke Risk Prediction". 
# Multimodal Contrastive Learning with Tabular and Imaging Data

Please cite our NeurIPS paper, [A Self-Supervised Model for Multi-modal Stroke Risk Prediction](link), if this code was helpful.

```
#put paper citation
```

If you want an overview of the paper, checkout:
- [graphical abstract?]
- 


## Instructions

Install environment using `conda env create --file environment.yaml`. 

To run, execute `python run.py`.

### Arguments - Command Line

If pretraining, pass `pretrain=True` and `datatype={imaging|multimodal|tabular}` for the desired pretraining type. `multimodal` uses our strategy from the paper, `tabular` uses SCARF, and `imaging` can be specified with the `loss` argument. Default is SimCLR with NTXEntLoss, other options includes BarlowTwins, which is still under development.

If you do not pass `pretrain=True`, the model will train fully supervised with the data modality specified in `datatype`: `tabular` or `imaging`.

You can evaluate a model by passing the path to the final pretraining checkpoint with the argument `checkpoint={PATH_TO_CKPT}`. After pretraining, a model will be evaluated with the default settings (frozen eval, lr=1e-3).

### Arguments - Hydra

All argument defaults can be set in hydra yaml files found in the configs folder.

Most arguments are set to those in the paper and work well out of the box. Default model is ResNet50.

Code is integrated with weights and biases, so set `wandb_project` and `wandb_entity` in [config.yaml](configs/config.yaml).

Path to folder containing data is set through the `data_base` argument and then joined with filenames set in the dataset yamls. 
- For the images, provide a .pt with a list of your images or a list of the paths to your images.
  - If providing a list of paths, set `live_loading=True`.
- If `weights` is set, during finetuning a weighted sampled will be used instead of assuming the evaluation train data has been properly balanced
- `eval_metric` supports `acc` for accuracy (top-1) and `auc` (for unbalanced data)
- If doing multimodal pretraining or tabular pretraining (SCARF), the tabular data should be provided as *NOT* one-hot encoded so the sampling from the empirical marginal distribution works correctly. You must provide a file `field_lengths_tabular` which is an array that in the order of your tabular columns specifies how many options there are for that field. Continuous fields should thus be set to 1 (i.e. no one-hot encoding necessary), while categorical fields should specify how many columns should be created for the one_hot encoding  

### Data

The UKBB data is semi-private. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access).

Should we add processing for UKBB splits/dataset ?
