# Loading Perturbation Data (Norman Dataset)

This guide explains how to load and use perturbation single-cell data in this project, specifically the Norman dataset.

## Overview

The Norman dataset contains single-cell RNA-seq data from CRISPR perturbation experiments. This integration allows you to:
- Load the Norman dataset from a repository cache
- Convert perturbation data into PyTorch-compatible formats
- Create train/val/test dataloaders for machine learning
- Handle perturbation labels automatically

## Quick Start

### Option 1: Simple Dataloader Creation (Recommended)

```python
from src.data.dataloader import get_perturbation_dataloaders

# Load Norman data with automatic train/val/test split
train_loader, val_loader, test_loader, metadata = get_perturbation_dataloaders(
    dataset_name="norman",
    batch_size=64,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_workers=0,  # Set to 0 on Windows
    seed=42
)

# Use in training
for features, labels in train_loader:
    # features: (batch_size, n_genes) - gene expression matrix
    # labels: (batch_size,) - encoded perturbation labels
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... training code
```

### Option 2: Direct Data Loading

```python
from src.data.perturbation_loader import load_norman_data, extract_features_and_labels
from src.data.dataset import GenomicDataset

# Load raw Norman data
norman_data = load_norman_data(save_dir="data/raw")

# Extract features and labels
features, labels, metadata = extract_features_and_labels(norman_data)

# Create dataset
dataset = GenomicDataset(
    features=features,
    labels=labels,
    data_source="array",
    label_encode=True
)
```

### Option 3: Using GenomicDataset Directly

```python
from src.data.dataset import GenomicDataset
from src.data.dataloader import get_dataloaders

# Create dataset that loads Norman data internally
dataset = GenomicDataset(
    data_path="norman",  # Dataset name
    data_source="perturbation",
    label_encode=True
)

# Create dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    dataset=dataset,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

## Integration with Training Scripts

### Modify train.py

You can now load Norman data in your training script:

```python
# In train.py
from src.data.dataloader import get_perturbation_dataloaders

def main(args):
    # Load perturbation data instead of regular files
    if args.data_source == "perturbation":
        train_loader, val_loader, test_loader, metadata = get_perturbation_dataloaders(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            num_workers=args.num_workers
        )
        
        # Get dimensions from metadata
        input_dim = metadata['n_features']
        num_classes = metadata['n_classes']
        
        logger.info(f"Loaded {metadata['dataset_name']} dataset")
        logger.info(f"Samples: {metadata['n_samples']}, Features: {input_dim}, Classes: {num_classes}")
    else:
        # Use existing file-based loading
        train_loader, val_loader, test_loader = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size
        )
```

## File Structure

The perturbation data loading is organized as follows:

```
src/data/
├── __init__.py
├── dataloader.py              # Main dataloader creation functions
├── dataset.py                 # GenomicDataset class (updated)
└── perturbation_loader.py     # Perturbation-specific loading utilities
```

### Key Components

1. **perturbation_loader.py**: Low-level utilities for loading perturbation data
   - `load_norman_data()`: Load Norman dataset from repository
   - `load_perturbation_data()`: General loader for any perturbation dataset
   - `extract_features_and_labels()`: Convert PertData to numpy arrays

2. **dataset.py**: PyTorch Dataset class
   - Updated `GenomicDataset` with `data_source="perturbation"` support
   - Automatic label encoding for perturbation names
   - Metadata storage for label mappings

3. **dataloader.py**: DataLoader creation
   - `get_perturbation_dataloaders()`: Convenience function for perturbation data
   - Updated `get_dataloaders()` with `data_source` parameter

## Data Format

### Input (PertData)
The raw data from `load_norman_data()` is a PertData object with:
- **X**: Gene expression matrix (cells × genes)
- **obs**: Cell metadata including perturbation labels
- **var**: Gene metadata

### Output (Processed)
After processing:
- **Features**: `(n_cells, n_genes)` float32 numpy array
- **Labels**: `(n_cells,)` int64 numpy array (encoded)
- **Metadata**: Dictionary with:
  - `n_cells`: Number of cells
  - `n_genes`: Number of genes
  - `label_mapping`: Perturbation name → integer
  - `inverse_label_mapping`: integer → Perturbation name
  - `gene_names`: List of gene names

## Label Encoding

Perturbation labels are automatically encoded:

```python
# Original labels (strings)
["CTRL", "CTRL", "ARID1A", "ARID1A", "BAZ1A", ...]

# Encoded labels (integers)
[0, 0, 1, 1, 2, ...]

# Label mapping (stored in metadata)
{
    "CTRL": 0,
    "ARID1A": 1,
    "BAZ1A": 2,
    ...
}
```

To decode predictions back to perturbation names:

```python
predictions = model(features).argmax(dim=1)  # Get predicted class indices

# Convert back to perturbation names
for pred_idx in predictions:
    pert_name = metadata['inverse_label_mapping'][pred_idx.item()]
    print(f"Predicted perturbation: {pert_name}")
```

## Requirements

Make sure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies for perturbation data:
- `scanpy>=1.9.0`: Single-cell data analysis
- `anndata>=0.9.0`: Annotated data matrices
- Standard ML libraries (torch, numpy, pandas)

## Module Dependencies

The code expects the perturbation data analysis module at:
```
src/exercises/perturbation_data_analysis/pertdata.py
```

This module should provide the `PertData` class with a `from_repo()` method.

## Example Usage

Run the provided example script to test the integration:

```bash
python load_norman_example.py
```

This will:
1. Load the Norman dataset
2. Create train/val/test dataloaders
3. Inspect a batch of data
4. Display dataset statistics and label mappings

## Customization

### Adding More Datasets

To support additional perturbation datasets (e.g., Dixit, Adamson):

1. Add a loader function in `perturbation_loader.py`:
```python
def load_dixit_data(save_dir="data/raw", force_reload=False):
    from src.exercises.perturbation_data_analysis import pertdata as pt
    git_root = get_git_root()
    save_path = git_root / save_dir
    dixit_data = pt.PertData.from_repo(name="dixit", save_dir=str(save_path))
    return dixit_data
```

2. Register it in the `load_perturbation_data()` function:
```python
dataset_loaders = {
    "norman": load_norman_data,
    "dixit": load_dixit_data,  # Add new dataset here
}
```

3. Use it:
```python
train_loader, val_loader, test_loader, metadata = get_perturbation_dataloaders(
    dataset_name="dixit",  # Specify new dataset
    batch_size=64
)
```

### Filtering Control Samples

To exclude control samples:

```python
from src.data.perturbation_loader import load_norman_data, extract_features_and_labels

norman_data = load_norman_data()
features, labels, metadata = extract_features_and_labels(
    norman_data,
    use_control=False  # Exclude control samples
)
```

### Custom Observation Keys

If your perturbation labels are stored under a different key:

```python
features, labels, metadata = extract_features_and_labels(
    pert_data,
    obs_key="your_custom_key"  # Default is "perturbation"
)
```

## Troubleshooting

### ImportError: Cannot import pertdata module

**Solution**: Ensure the perturbation data analysis module exists at:
```
src/exercises/perturbation_data_analysis/pertdata.py
```

### Git root not found

**Solution**: Make sure you're running from within a git repository. The code uses `git rev-parse --show-toplevel` to find the project root.

### Data not downloading

**Solution**: Check your internet connection and ensure the `data/raw` directory is writable.

### Windows multiprocessing issues

**Solution**: Set `num_workers=0` in dataloader functions when on Windows:
```python
train_loader, val_loader, test_loader, metadata = get_perturbation_dataloaders(
    dataset_name="norman",
    num_workers=0  # Important for Windows
)
```

## Performance Tips

1. **Batch size**: Start with 32-64 and adjust based on GPU memory
2. **Number of workers**: Use 0 on Windows, 4-8 on Linux/Mac
3. **Data caching**: Data is cached after first download in `data/raw/`
4. **Memory**: Gene expression matrices can be large; monitor RAM usage

## Next Steps

1. Modify your model architecture to match the feature dimension
2. Update training scripts to use `get_perturbation_dataloaders()`
3. Experiment with different train/val/test splits
4. Add data augmentation for robustness
5. Try different perturbation datasets

## References

- Norman et al. (2019). "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes." Science.
