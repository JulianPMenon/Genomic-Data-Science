"""
Genomic ML - Machine Learning for Genomic Data Science
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for convenience
from src.models.base_model import BaseModel
from src.data.dataset import GenomicDataset
from src.data.dataloader import get_dataloaders, get_single_dataloader
from src.utils.logger import setup_logger
from src.utils.config import load_config, save_config, Config
from src.utils.metrics import calculate_metrics, print_metrics

__all__ = [
    "BaseModel",
    "GenomicDataset",
    "get_dataloaders",
    "get_single_dataloader",
    "setup_logger",
    "load_config",
    "save_config",
    "Config",
    "calculate_metrics",
    "print_metrics",
]
