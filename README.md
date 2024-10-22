# SSMSRPM
GitHub repository for the code of the project and paper "A Self-Supervised Model for Multi-modal Stroke Risk Prediction". 
# A Self-Supervised Multi-modal Stroke Risk Prediction Model

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

If pretraining, pass `pretrain=True` and `datatype={imaging|multimodal|tabular}` for the desired pretraining type. `multimodal` uses the strategy from the paper, `tabular` uses SCARF, and `imaging` can be specified. For imaging, the default method is SimCLR with the NTXEntLoss, and other options includes BarlowTwins, which is still under development. For tabular, the default method is SCARF. For multimodal pre-training, one can choose between MMCL or the current strategy (SSMSRPM). 

If you do not pass `pretrain=True`, the model will train fully supervised with the data modality specified in `datatype`: `tabular` or `imaging`.

You can evaluate a model by passing the path to the final pretraining checkpoint with the argument `checkpoint={PATH_TO_CKPT}`. After pretraining, a model will be evaluated with the default settings in the config file, to be adapted according to the algorithm_name, the strategy used, as well as the datatype (or eval_datatype for the fine-tuning). Our best checkpoint obtained can be used from this link: 

### Arguments - Hydra

All argument defaults can be set in hydra yaml files found in the configs folder.

For reference to typical arguments, see the default config files for pretraining and finetuning. Don't hesitate to experiment with different parameter configurations. Default model is ResNet50.

Code is integrated with weights and biases, so set `wandb_project` and `wandb_entity` in [config_pretraining.yaml](configs/config_pretraining.yaml).

Paths to your data is set through the `data_base` argument and then joined with filenames set in the dataset yamls. Therefore, you have to modify the config file in : SSMSRPM\configs\dataset\ukb_stroke.yaml to your own paths containing the data.

- For the images, a list of the paths to your images in .pt format.
- If `stratified_sampler` is set, during finetuning a stratified sampler will be used to balance the training batches.
- `eval_metric` supports `auc`, `bAcc`, `F1` and `Recall` (sensitivity).
- If doing multimodal pretraining or tabular pretraining (SCARF), the tabular data should be provided as *NOT* one-hot encoded so the sampling from the empirical marginal distribution works correctly. You must provide a file `field_lengths_tabular` which is an array that in the order of your tabular columns specifies how many options there are for that field. Continuous fields should thus be set to 1 (i.e. no one-hot encoding necessary), while categorical fields should specify how many columns should be created for the one_hot encoding  

### Data

The UKBB data is semi-private. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access).
You must therefore provide tabular features pre-processed for missing variables and standardized, as well as T2-FLAIR anatomical brain MRI scans that are defaced, anonymized and deskulled, according to the UKB's general pre-processing pipeline [UKB pipeline](https://www.fmrib.ox.ac.uk/ukbiobank/fbp/). 
