"""
Dataset class for loading genomic data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class GenomicDataset(Dataset):
    """
    Custom Dataset for genomic data.
    
    This is a template that can be adapted for various genomic data formats.
    Examples: gene expression data, DNA sequences, protein sequences, etc.
    """
    
    def __init__(self, data_path=None, features=None, labels=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str, optional): Path to data file (CSV, TSV, etc.)
            features (np.ndarray, optional): Feature matrix
            labels (np.ndarray, optional): Label vector
            transform (callable, optional): Optional transform to apply to samples
        """
        self.transform = transform
        
        if data_path is not None:
            # Load data from file
            self._load_from_file(data_path)
        elif features is not None and labels is not None:
            # Use provided features and labels
            self.features = features
            self.labels = labels
        else:
            # Create dummy data for demonstration
            self._create_dummy_data()
    
    def _load_from_file(self, data_path):
        """
        Load data from a file.
        
        Args:
            data_path (str): Path to data file
        """
        # Example: Load CSV file with pandas
        df = pd.read_csv(data_path)
        
        # Assuming last column is the label
        self.features = df.iloc[:, :-1].values
        self.labels = df.iloc[:, -1].values
    
    def _create_dummy_data(self):
        """
        Create dummy data for demonstration purposes.
        """
        # Create random data
        n_samples = 1000
        n_features = 100
        
        self.features = np.random.randn(n_samples, n_features).astype(np.float32)
        self.labels = np.random.randint(0, 2, n_samples).astype(np.int64)
    
    def __len__(self):
        """
        Return the total number of samples.
        
        Returns:
            int: Number of samples
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (features, label) for the sample
        """
        features = self.features[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_feature_dim(self):
        """
        Get the feature dimension.
        
        Returns:
            int: Feature dimension
        """
        return self.features.shape[1] if len(self.features.shape) > 1 else 1
    
    def get_num_classes(self):
        """
        Get the number of unique classes.
        
        Returns:
            int: Number of classes
        """
        return len(np.unique(self.labels))
