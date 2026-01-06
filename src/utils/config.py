"""
Configuration management utilities
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save YAML config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class Config:
    """
    Configuration class for easy attribute access.
    """
    
    def __init__(self, config_dict):
        """
        Initialize config from dictionary.
        
        Args:
            config_dict (dict): Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return str(self.__dict__)
