"""
Dataset class for loading genomic data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class GenomicDataset(Dataset):
    """
    Custom Dataset for genomic data.
    
    This is a template that can be adapted for various genomic data formats.
    Examples: gene expression data, DNA sequences, protein sequences, 
    perturbation data, etc.
    """
    
    def __init__(self, data_path=None, features=None, labels=None, 
                 transform=None, data_source="file", label_encode=True):
        """
        Initialize the dataset.
        
        Args:
            data_path (str, optional): Path to data file (CSV, TSV, etc.)
            features (np.ndarray, optional): Feature matrix
            labels (np.ndarray, optional): Label vector
            transform (callable, optional): Optional transform to apply to samples
            data_source (str): Type of data source ("file", "array", "perturbation", "dummy")
            label_encode (bool): Whether to encode string labels to integers
        """
        self.transform = transform
        self.label_encode = label_encode
        self.label_encoder = None
        self.metadata = {}
        
        if data_source == "perturbation":
            # Load perturbation data (e.g., Norman dataset)
            self._load_perturbation_data(data_path)
        elif data_path is not None:
            # Load data from file
            self._load_from_file(data_path)
        elif features is not None and labels is not None:
            # Use provided features and labels
            self.features = features
            self.labels = labels
            if label_encode and labels.dtype.kind in ['U', 'O', 'S']:  # String types
                self._encode_labels()
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
    
    def _load_perturbation_data(self, dataset_name="norman"):
        """
        Load perturbation data (e.g., Norman dataset).
        
        Args:
            dataset_name (str): Name of the perturbation dataset to load
        """
        from .perturbation_loader import (
            load_perturbation_data,
            extract_features_and_labels
        )
        
        # Load the perturbation data
        pert_data = load_perturbation_data(dataset_name=dataset_name)
        
        # Extract features and labels
        features, labels, metadata = extract_features_and_labels(pert_data)
        
        # Store as float32 for efficiency
        self.features = features.astype(np.float32)
        self.labels = labels
        self.metadata = metadata
        
        # Encode string labels to integers if needed
        if self.label_encode and labels.dtype.kind in ['U', 'O', 'S']:
            self._encode_labels()
    
    def _encode_labels(self):
        """
        Encode string labels to integers using LabelEncoder.
        """
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels).astype(np.int64)
        self.metadata['label_mapping'] = dict(
            zip(self.label_encoder.classes_, 
                self.label_encoder.transform(self.label_encoder.classes_))
        )
        self.metadata['inverse_label_mapping'] = dict(
            enumerate(self.label_encoder.classes_)
        )
    
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
