"""
Base Model - Template for PyTorch models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base neural network model template.
    
    This is a simple example model that can be extended for specific tasks.
    For genomic data, you might want to use:
    - CNN for sequence data
    - RNN/LSTM for sequential patterns
    - Transformer for complex relationships
    - Graph Neural Networks for interaction networks
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes for classification)
            dropout (float): Dropout rate for regularization
        """
        super(BaseModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_num_params(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """
        Get the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
