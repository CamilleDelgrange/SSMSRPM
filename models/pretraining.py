from typing import List, Tuple, Dict, Any

import torch
#torch.cuda.empty_cache()
import pytorch_lightning as pl
import torchmetrics
import torchvision
from sklearn.linear_model import LogisticRegression
from lightly.models.modules import SimCLRProjectionHead
from models.ResnetEmbeddingModel import ProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.Tip_utils.Transformer import TabularTransformerEncoder, MultimodalTransformerEncoder, TabularPredictor
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from monai.networks import nets
from torch import nn
import torch.nn.functional as F
from models.TabularEncoder_ITM import TabularEncoder


# Disable TF32 on matmul operations.
#torch.backends.cuda.matmul.allow_tf32 = False
# Disable TF32 on cudnn operations.
#torch.backends.cudnn.allow_tf32 = False

# Verify that TF32 is disabled
#print(f"TF32 on matmul operations: {torch.backends.cuda.matmul.allow_tf32}")
#print(f"TF32 on cudnn operations: {torch.backends.cudnn.allow_tf32}")

class Pretraining(pl.LightningModule):
    
  def __init__(self, hparams) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)
  
  def monai_3d_ssl_encoder(self, model_name='resnet50', pretrained=True):
    if model_name == 'resnet18':
        model = nets.resnet18(spatial_dims=3, n_input_channels=1)
    elif model_name == 'resnet34':
        model = nets.resnet34(spatial_dims=3, n_input_channels=1)
    elif model_name == 'resnet50':
        model = nets.resnet50(spatial_dims=3, n_input_channels=1)
    elif model_name == 'resnet101':
        model = nets.resnet101(spatial_dims=3, n_input_channels=1)
    elif model_name == 'resnet152':
        model = nets.resnet152(spatial_dims=3, n_input_channels=1)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # Remove the final fully connected layer to get only the feature extractor
    modules = list(model.children())[:-2] # remove classifier layer + adaptive pool avg
    model = nn.Sequential(*modules)

    return model

  def initialize_multimodal_encoder_and_predictor(self) -> None:
    self.encoder_multimodal = MultimodalTransformerEncoder(self.hparams)
   
  def initialize_imaging_encoder_and_projector(self) -> None:
    """
    Selects appropriate resnet encoder
    """
    model = self.monai_3d_ssl_encoder(self.hparams.model)
    self.encoder_imaging = model
    #print("Params", self.hparams.model)
    self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
    print("Embedding dim", self.hparams.embedding_dim)
    self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim) # input_dim, hidden_dim, output_dim, num_layers, batch_norm

  def initialize_tabular_encoder_and_projector(self) -> None:
    self.encoder_tabular = TabularEncoder(self.hparams)
    self.projector_tabular = SimCLRProjectionHead(self.hparams.tabular_embedding_dim, self.hparams.tabular_embedding_dim, self.hparams.projection_dim)

  def initialize_classifier_and_metrics(self, nclasses_train, nclasses_val):
    """
    Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
    """
    # Classifier
    self.estimator = None

    # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
    self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train)
    self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

    self.classifier_auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    
    if self.hparams.hard_neg: 
      self.acc_train_itm = torchmetrics.Accuracy(task='binary', num_classes=2)
      self.acc_val_itm = torchmetrics.Accuracy(task='binary', num_classes=2)


  def load_pretrained_imaging_weights(self) -> None:
    """
    Can load imaging encoder with pretrained weights from previous checkpoint/run
    """
    loaded_chkpt = torch.load(self.hparams.imaging_pretrain_checkpoint)
    state_dict = loaded_chkpt['state_dict']
    state_dict_encoder = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
    _ = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=True)
    print("Loaded imaging weights")
    if self.hparams.pretrained_imaging_strategy == 'frozen':
      for _, param in self.encoder_imaging.named_parameters():
        param.requires_grad = False
      parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
      assert len(parameters)==0

  def forward_multimodal_feature(self, tabular_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
    """
    Generates multimodal feature from transformer integration.
    """
    y = self.encoder_multimodal(x=tabular_features, image_features=image_features)
    return y[:,0,:]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates encoding of imaging data.
    """
    z, y, y_flatt = self.forward_imaging(x)
    return y

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of imaging data.
    """
    #print(x.shape)
    #print(x.is_contiguous())
    y = self.encoder_imaging(x) #[0]
    #print("output encoder", y.shape)
    #print("output encoder v2", self.encoder_imaging(x)[0].shape)
    y_flatt = F.adaptive_avg_pool3d(y, (1, 1, 1)).flatten(1) #y.view(y.size(0), -1)  # Flatten the output tensor
    #print("flattened", y.shape)
    z = self.projector_imaging(y_flatt)
    #print("projected", z.shape)
    return z, y, y_flatt

  def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of tabular data.
    """
    y = self.encoder_tabular(x)
    y_flatt = y.flatten(start_dim=1)
    z = self.projector_tabular(y_flatt)
    return z, y, y_flatt
  
  def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_train(logits, labels)
    
    self.log(f"{modality}.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)

  def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_val(logits, labels)
    
    self.log(f"{modality}.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
  
  def calc_and_log_train_itm_acc(self, logits, labels, modality: str) -> None:
    logits, labels = logits.detach(), labels.detach()
    self.acc_train_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
    self.log(f"{modality}.train.ITMacc", self.acc_train_itm, on_epoch=True, on_step=False)
  
  def calc_and_log_val_itm_acc(self, logits, labels, modality: str) -> None:
    logits, labels = logits.detach(), labels.detach()
    self.acc_val_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
    self.log(f"{modality}.val.ITMacc", self.acc_val_itm, on_epoch=True, on_step=False)

  def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
    """
    Train and log classifier
    """
    if self.current_epoch != 0 and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(train_step_outputs)
    
      self.estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(embeddings, labels) #solver='newton_cholesky'
      preds, probs = self.predict_live_estimator(embeddings)

      self.classifier_auc_train(probs, labels)

      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)
      del embeddings, labels
      torch.cuda.empty_cache()

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step and compute validation classifier performance
    """
    if self.hparams.log_images:
      print(self.hparams.log_images)
      print("Validation step outputs", validation_step_outputs)
      print("Example image", [validation_step_outputs[0]['sample_augmentation']])
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

    # Validate classifier
    if not self.estimator is None and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(validation_step_outputs)
      
      preds, probs = self.predict_live_estimator(embeddings)
      
      self.classifier_auc_val(probs, labels)

      self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)
      del embeddings, labels
      torch.cuda.empty_cache()
      


  def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack outputs from multiple steps
    """
    labels = outputs[0]['labels']
    embeddings = outputs[0]['embeddings']
    for i in range(1, len(outputs)):
      labels = torch.cat((labels, outputs[i]['labels']), dim=0)
      embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

    embeddings = embeddings.detach().cpu()
    labels = labels.cpu()

    return embeddings, labels

  def predict_live_estimator(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict using live estimator
    """
    preds = self.estimator.predict(embeddings)
    probs = self.estimator.predict_proba(embeddings)

    preds = torch.tensor(preds)
    probs = torch.tensor(probs)
    
    # Only need probs for positive class in binary case
    if self.hparams.num_classes == 2:
      probs = probs[:,1]

    return preds, probs


  def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    return scheduler
  