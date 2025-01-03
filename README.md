# SSMSRPM
GitHub repository for the code of the project and paper "A Self-Supervised Model for Multi-modal Stroke Risk Prediction". 

# A Self-Supervised Multi-modal Stroke Risk Prediction Model

Please cite our NeurIPS paper, [A Self-Supervised Model for Multi-modal Stroke Risk Prediction](https://arxiv.org/abs/2411.09822), if this code was helpful.

```
@inproceedings{
delgrange2024a,
title={A Self-Supervised Model for Multi-modal Stroke Risk Prediction},
author={Camille Delgrange and Olga V. Demler and Samia Mora and bjoern menze and Ezequiel De la Rosa and Neda Davoudi},
booktitle={Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond},
year={2024},
url={https://openreview.net/forum?id=ST72dbOvwx}
}
```

![Alt Text](Diagram_paper.png)

## Instructions

Run the following commands to install and activate the environment. Then install any needed additional packages.

```
conda env create --file environment.yaml
conda activate selfsuper
pip install torch-geometric
pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

To run, execute `python run.py`.

### Arguments - Command Line

If pretraining, pass `pretrain=True` and `datatype={imaging|multimodal|tabular}` for the desired pretraining type. `multimodal` uses the strategy from the paper, `tabular` uses SCARF, and `imaging` can be specified. For imaging, the default method is SimCLR with the NTXEntLoss, and other options includes BarlowTwins, which is still under development. For tabular, the default method is SCARF. For multimodal pre-training, one can choose between MMCL or the current strategy (SSMSRPM). 

If you do not pass `pretrain=True`, the model will train fully supervised with the data modality specified in `datatype`: `tabular` or `imaging`.

You can evaluate a model by passing the path to the final pretraining checkpoint with the argument `checkpoint={PATH_TO_CKPT}`. After pretraining, a model will be evaluated with the default settings in the config file, to be adapted according to the algorithm_name, the strategy used, as well as the datatype (or eval_datatype for the fine-tuning). The best model's weights and checkpoint can be downloaded from the links below.

### Arguments - Hydra

All argument defaults can be set in hydra yaml files found in the configs folder.

For reference to typical arguments, see the default config files for pretraining and finetuning. Don't hesitate to experiment with different parameter configurations. Default model is ResNet50.

Code is integrated with weights and biases, so set `wandb_project` and `wandb_entity` in [config_pretraining.yaml](configs/config_pretraining.yaml).

Paths to your data is set through the `data_base` argument and then joined with filenames set in the dataset yaml files. Therefore, you have to modify the config file in : SSMSRPM/configs/dataset/[dataset].yaml to your own paths containing the data.

- For the images, provide a list of the paths to your images in .pt format. The images can be 3D anatomical MRI volumes or segmented lesion masks available from the UK Biobank pre-processing pipeline depending on the strategy. 
- If `stratified_sampler` is set, during finetuning a stratified sampler will be used to balance the training batches.
- `eval_metric` supports `auc`, `bAcc`, `F1` and `Recall` (sensitivity).
- For tabular data and labels, files are in .csv format.
- If doing multimodal pretraining or tabular pretraining (SCARF), the tabular data should be provided as *NOT* one-hot encoded so the sampling from the empirical marginal distribution works correctly. You must provide a file `field_lengths_tabular` which is an array that in the order of your tabular columns specifies how many options there are for that field. Continuous fields should thus be set to 1 (i.e. no one-hot encoding necessary), while categorical fields should specify how many columns should be created for the one_hot encoding.

### Model Weights

You can download our best model's weights in `.pth` and `.ckpt` format from the following Google Drive link:

[Download Best Model's Weights](https://drive.google.com/file/d/1_XIFjztzUBEPOnnM45Gpl69_-Lw4Kb5P/view?usp=sharing)
[Download Best Model's Checkpoint](https://drive.google.com/file/d/1dj1H1-QzX0tvTuY6e4tFSv_mu4CfyvdO/view?usp=sharing)

### Data

The UKBB data is semi-private. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access).
You must therefore provide tabular features pre-processed for missing variables and standardized, as well as T2-FLAIR anatomical brain MRI scans that are defaced, anonymized and deskulled, according to the UKB's general pre-processing pipeline [UKB pipeline](https://www.fmrib.ox.ac.uk/ukbiobank/fbp/). 

### Acknowledgments:

We thank the public repositories of [TIP](https://github.com/siyi-wind/TIP/tree/main) and [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging/tree/65070559e0a59944c4c78c667fc6372e8828520e), that helped us build this project and adapt our code for 3D multimodal data and our clinical interpretation of the task. 
