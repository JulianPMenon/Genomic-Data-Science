"""
Perturbation data loader for single-cell perturbation datasets.
This module provides utilities to load and process perturbation data
such as the Norman dataset using the perturbation analysis library.
"""

import os
import sys
from pathlib import Path


def get_git_root():
    """
    Get the root directory of the git repository.
    
    Returns:
        Path: Path object pointing to the git root directory
    """
    try:
        git_root = os.popen(cmd="git rev-parse --show-toplevel").read().strip()
        return Path(git_root)
    except Exception as e:
        # Fallback: try to find the root by looking for .git directory
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        raise RuntimeError(f"Could not find git root: {e}")


def setup_pertdata_path():
    """
    Add the git root to Python path to enable imports from the repository.
    This is necessary for importing the perturbation data analysis module.
    """
    git_root = get_git_root()
    if str(git_root) not in sys.path:
        sys.path.insert(0, str(git_root))


def load_norman_data(save_dir="data/raw", force_reload=False):
    """
    Load the Norman perturbation dataset.
    
    This function loads the Norman dataset from the repository or cache.
    The Norman dataset contains single-cell perturbation data from:
    Norman et al. (2019) Science - Exploring genetic interaction manifolds 
    constructed from rich single-cell phenotypes.
    
    Args:
        save_dir (str): Directory to save/load the data. Relative to project root.
        force_reload (bool): If True, force re-download even if cached.
        
    Returns:
        PertData: Loaded perturbation data object with gene expression and metadata
        
    Raises:
        ImportError: If the perturbation data analysis module is not available
        RuntimeError: If data loading fails
    """
    # Setup path for imports
    setup_pertdata_path()
    
    try:
        from src.exercises.perturbation_data_analysis import pertdata as pt
    except ImportError:
        raise ImportError(
            "Could not import perturbation data analysis module. "
            "Please ensure the module is available at "
            "'src/exercises/perturbation_data_analysis/pertdata.py'"
        )
    
    # Convert save_dir to absolute path relative to git root
    git_root = get_git_root()
    save_path = git_root / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the Norman dataset
        norman_data = pt.PertData.from_repo(
            name="norman",
            save_dir=str(save_path)
        )
        return norman_data
    except Exception as e:
        raise RuntimeError(f"Failed to load Norman dataset: {e}")


def load_perturbation_data(dataset_name="norman", save_dir="data/raw", **kwargs):
    """
    General function to load perturbation datasets.
    
    Args:
        dataset_name (str): Name of the dataset to load (e.g., "norman", "dixit", etc.)
        save_dir (str): Directory to save/load the data
        **kwargs: Additional arguments passed to the loading function
        
    Returns:
        PertData: Loaded perturbation data object
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_loaders = {
        "norman": load_norman_data,
        # Add more datasets here as needed
        # "dixit": load_dixit_data,
        # "adamson": load_adamson_data,
    }
    
    if dataset_name.lower() not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(dataset_loaders.keys())}"
        )
    
    loader_func = dataset_loaders[dataset_name.lower()]
    return loader_func(save_dir=save_dir, **kwargs)


def extract_features_and_labels(pert_data, obs_key="perturbation", use_control=True):
    """
    Extract features (gene expression) and labels (perturbation) from PertData object.
    
    Args:
        pert_data: PertData object containing the perturbation experiment data
        obs_key (str): Key in the observations (obs) containing perturbation labels
        use_control (bool): Whether to include control samples
        
    Returns:
        tuple: (features, labels, metadata)
            - features: numpy array of shape (n_cells, n_genes) with gene expression
            - labels: numpy array of shape (n_cells,) with perturbation labels
            - metadata: dict with additional information (gene names, etc.)
    """
    # Extract gene expression matrix (cells x genes)
    # Assuming pert_data has an AnnData structure with X attribute
    if hasattr(pert_data, 'X'):
        features = pert_data.X
    elif hasattr(pert_data, 'adata') and hasattr(pert_data.adata, 'X'):
        features = pert_data.adata.X
    else:
        raise AttributeError("Could not find expression matrix in PertData object")
    
    # Convert sparse matrix to dense if needed
    if hasattr(features, 'toarray'):
        features = features.toarray()
    
    # Extract perturbation labels
    if hasattr(pert_data, 'obs'):
        labels = pert_data.obs[obs_key].values
    elif hasattr(pert_data, 'adata') and hasattr(pert_data.adata, 'obs'):
        labels = pert_data.adata.obs[obs_key].values
    else:
        raise AttributeError("Could not find observation metadata in PertData object")
    
    # Filter out control if needed
    if not use_control:
        control_mask = labels != "control"
        features = features[control_mask]
        labels = labels[control_mask]
    
    # Prepare metadata
    metadata = {
        "n_cells": features.shape[0],
        "n_genes": features.shape[1],
        "obs_key": obs_key,
    }
    
    # Add gene names if available
    if hasattr(pert_data, 'var_names'):
        metadata["gene_names"] = pert_data.var_names
    elif hasattr(pert_data, 'adata') and hasattr(pert_data.adata, 'var_names'):
        metadata["gene_names"] = pert_data.adata.var_names
    
    return features, labels, metadata
