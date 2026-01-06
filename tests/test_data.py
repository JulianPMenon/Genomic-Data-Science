"""
Unit tests for data loading utilities
"""

import pytest
import torch
from src.data.dataset import GenomicDataset
from src.data.dataloader import get_dataloaders, get_single_dataloader


def test_dataset_initialization():
    """Test dataset initialization."""
    dataset = GenomicDataset()
    assert len(dataset) > 0


def test_dataset_getitem():
    """Test getting items from dataset."""
    dataset = GenomicDataset()
    features, label = dataset[0]
    
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert features.dtype == torch.float32
    assert label.dtype == torch.long


def test_dataset_properties():
    """Test dataset properties."""
    dataset = GenomicDataset()
    
    feature_dim = dataset.get_feature_dim()
    num_classes = dataset.get_num_classes()
    
    assert feature_dim > 0
    assert num_classes > 0


def test_get_dataloaders():
    """Test dataloader creation."""
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,  # Use 0 for testing
        seed=42
    )
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_dataloader_batch_shape():
    """Test batch shape from dataloader."""
    dataset = GenomicDataset()
    loader = get_single_dataloader(dataset, batch_size=16, num_workers=0)
    
    features, labels = next(iter(loader))
    
    assert features.shape[0] <= 16  # Batch size
    assert labels.shape[0] <= 16


def test_dataloader_splits_sum_to_total():
    """Test that train/val/test splits sum to dataset size."""
    dataset = GenomicDataset()
    total_size = len(dataset)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,
        seed=42
    )
    
    # Calculate total samples in all loaders
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    
    assert train_samples + val_samples + test_samples == total_size
