"""
Unit tests for the BaseModel
"""

import pytest
import torch
from src.models.base_model import BaseModel


def test_model_initialization():
    """Test that model initializes correctly."""
    model = BaseModel(input_dim=100, hidden_dim=256, output_dim=2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    """Test forward pass with dummy data."""
    model = BaseModel(input_dim=100, hidden_dim=256, output_dim=2)
    batch_size = 32
    input_data = torch.randn(batch_size, 100)
    
    output = model(input_data)
    
    assert output.shape == (batch_size, 2)


def test_model_parameters():
    """Test parameter counting."""
    model = BaseModel(input_dim=100, hidden_dim=256, output_dim=2)
    
    num_params = model.get_num_params()
    trainable_params = model.get_trainable_params()
    
    assert num_params > 0
    assert trainable_params == num_params
    assert isinstance(num_params, int)


def test_model_device_transfer():
    """Test model can be moved to different devices."""
    model = BaseModel(input_dim=100, hidden_dim=256, output_dim=2)
    
    # Test CPU
    model = model.to('cpu')
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        assert next(model.parameters()).device.type == 'cuda'


def test_model_output_shape():
    """Test various input shapes."""
    model = BaseModel(input_dim=50, hidden_dim=128, output_dim=3)
    
    # Single sample
    single_input = torch.randn(1, 50)
    single_output = model(single_input)
    assert single_output.shape == (1, 3)
    
    # Batch
    batch_input = torch.randn(64, 50)
    batch_output = model(batch_input)
    assert batch_output.shape == (64, 3)
