"""
Unit tests for utility functions
"""

import pytest
import tempfile
import logging
from pathlib import Path
import numpy as np
import yaml

from src.utils.logger import setup_logger
from src.utils.config import load_config, save_config, Config
from src.utils.metrics import calculate_metrics


def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger(name="test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_logger_with_file():
    """Test logger with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logger(name="test_logger_file", log_file=str(log_file))
        
        logger.info("Test message")
        
        assert log_file.exists()


def test_config_save_and_load():
    """Test configuration save and load."""
    test_config = {
        'model': {
            'input_dim': 100,
            'hidden_dim': 256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        
        # Save config
        save_config(test_config, str(config_file))
        assert config_file.exists()
        
        # Load config
        loaded_config = load_config(str(config_file))
        assert loaded_config == test_config


def test_config_class():
    """Test Config class for attribute access."""
    config_dict = {
        'model': {'input_dim': 100},
        'training': {'batch_size': 32}
    }
    
    config = Config(config_dict)
    
    assert config.model.input_dim == 100
    assert config.training.batch_size == 32


def test_calculate_metrics_binary():
    """Test metrics calculation for binary classification."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.9])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, average='binary')
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    assert 'confusion_matrix' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1


def test_calculate_metrics_with_tensors():
    """Test metrics calculation with PyTorch tensors."""
    import torch
    
    y_true = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = torch.tensor([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.9])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, average='binary')
    
    assert 'accuracy' in metrics
    assert isinstance(metrics['accuracy'], float)
