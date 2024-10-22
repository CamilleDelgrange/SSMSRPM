from typing import List, Tuple
import random
import csv
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class ContrastiveImagingAndTabularDatasetCached(Dataset):
  """
    Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

    The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
    The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
    with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, data_path_imaging: str, data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool, labels_path: str) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging) # pt file in both cases

    # Tabular
    print(data_path_tabular)
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    print(len(self.data_tabular))
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    #print("tab lengths", self.field_lengths_tabular, len(self.field_lengths_tabular), torch.sum(torch.tensor(self.field_lengths_tabular)))
    self.one_hot_tabular = one_hot_tabular
    
    # Classifier
    self.labels = torch.load(labels_path)
    # Logging the length of lists
    print(f"Length of data: {len(self.data_imaging)}")
    print(f"Length of labels: {len(self.labels)}")
    print(f"Length of data_tabular: {len(self.data_tabular)}")

  def read_and_parse_csv(self, data_path_tabular: str) -> List[List[float]]:
      """
      Does what it says on the box.
      """
      with open(data_path_tabular, 'r') as f:
          reader = csv.reader(f)
          data = []
          #original_data_size = 0
          #num_columns = None
          #first_row_skipped = False  # Flag to track whether the first row has been encountered
          for r in reader:
              #if not first_row_skipped:
                  #first_row_skipped = True
                  #continue  # Skip the first row
              #if num_columns is None:
                  #num_columns = len(r)
                  #print("Number of columns in the original data:", num_columns)
              r2 = [float(r1) for r1 in r]
              #original_data_size += 1
              data.append(r2)
      #print("Original data size:", original_data_size)
      return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path) # don't call with header=None
    #print(data_path)
    #print(data_df)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
        return int(sum(self.field_lengths_tabular))
    else:
        return len(self.field_lengths_tabular)

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
        subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
        if self.field_lengths_tabular[i] == 1:
            out.append(subject[i].unsqueeze(0))
        else:
            value = subject[i].long()
            num_classes = int(self.field_lengths_tabular[i])
            if value >= num_classes:
                print(f"Value {value} at index {i} exceeds num_classes {num_classes}, capping to {num_classes - 1}")
                value = num_classes - 1
                value = torch.tensor([value], dtype=torch.long)  # Convert to tensor
            out.append(torch.nn.functional.one_hot(value, num_classes=num_classes))
            #print(int(self.field_lengths_tabular[i]))
    #print(f"Concatenated shape: {torch.cat(out).shape}")
    return torch.cat(out, dim=0)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    image_paths = self.data_imaging[index]
    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    #tabular_views = [tv.to("cuda:0") for tv in tabular_views]
    #print("Device for tab views", [tv.device for tv in tabular_views])
    #print(f"Shape of tabular views before: {[tv.shape for tv in tabular_views]}")
    if self.one_hot_tabular:
        tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
        #print("One hot", tabular_views)
    label = torch.tensor(self.labels[index], dtype=torch.long)
    #label = label.to("cuda:0")
    #print("Dev label", label.device)
    #print(f"Shape of tabular views: {[tv.shape for tv in tabular_views]}")
    return {
            'image': image_paths,
            'label': label,
            'tabular_views': tabular_views,
    }

  def __len__(self):
      return len(self.data_tabular)