"""
DataLoader utilities for creating train/val/test splits
"""

import torch
from torch.utils.data import DataLoader, random_split
from .dataset import GenomicDataset


def get_dataloaders(
    dataset=None,
    data_path=None,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_workers=4,
    seed=42
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset (Dataset, optional): PyTorch Dataset object
        data_path (str, optional): Path to data file (if dataset not provided)
        batch_size (int): Batch size for dataloaders
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create dataset if not provided
    if dataset is None:
        dataset = GenomicDataset(data_path=data_path)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate split sizes ensuring all samples are allocated
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    # Calculate remaining samples for test set to ensure no samples are lost due to rounding
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_single_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a single dataloader (useful for inference).
    
    Args:
        dataset (Dataset): PyTorch Dataset object
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of workers
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
