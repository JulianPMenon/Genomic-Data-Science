"""
Example script demonstrating how to load and use perturbation data (Norman dataset).

This script shows the basic workflow for loading the Norman single-cell perturbation
dataset and creating dataloaders for training machine learning models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import get_perturbation_dataloaders
from src.data.perturbation_loader import load_norman_data, extract_features_and_labels


def example_basic_loading():
    """
    Example 1: Basic loading of Norman perturbation data.
    """
    print("=" * 60)
    print("Example 1: Basic Norman dataset loading")
    print("=" * 60)
    
    # Load Norman data directly
    print("\nLoading Norman dataset...")
    norman_data = load_norman_data(save_dir="data/raw")
    print(f"✓ Norman data loaded successfully!")
    
    # Extract features and labels
    features, labels, metadata = extract_features_and_labels(norman_data)
    print(f"\nDataset statistics:")
    print(f"  - Number of cells: {metadata['n_cells']}")
    print(f"  - Number of genes: {metadata['n_genes']}")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Unique perturbations: {len(set(labels))}")
    
    return norman_data


def example_dataloader_creation():
    """
    Example 2: Create train/val/test dataloaders from Norman data.
    """
    print("\n" + "=" * 60)
    print("Example 2: Creating dataloaders for training")
    print("=" * 60)
    
    # Create dataloaders with automatic train/val/test split
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, metadata = get_perturbation_dataloaders(
        dataset_name="norman",
        batch_size=64,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,  # Set to 0 for Windows compatibility
        seed=42
    )
    
    print(f"✓ Dataloaders created successfully!")
    print(f"\nDataloader statistics:")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print(f"  - Batch size: {train_loader.batch_size}")
    print(f"  - Total samples: {metadata['n_samples']}")
    print(f"  - Feature dimension: {metadata['n_features']}")
    print(f"  - Number of classes: {metadata['n_classes']}")
    
    # Show label mapping if available
    if 'label_mapping' in metadata:
        print(f"\nPerturbation label mapping (first 10):")
        for i, (pert, idx) in enumerate(list(metadata['label_mapping'].items())[:10]):
            print(f"  {pert} -> {idx}")
        if len(metadata['label_mapping']) > 10:
            print(f"  ... and {len(metadata['label_mapping']) - 10} more")
    
    return train_loader, val_loader, test_loader, metadata


def example_batch_inspection():
    """
    Example 3: Inspect a batch of data from the dataloader.
    """
    print("\n" + "=" * 60)
    print("Example 3: Inspecting a batch of data")
    print("=" * 60)
    
    # Create dataloader
    train_loader, _, _, metadata = get_perturbation_dataloaders(
        dataset_name="norman",
        batch_size=32,
        num_workers=0
    )
    
    # Get first batch
    features_batch, labels_batch = next(iter(train_loader))
    
    print(f"\nBatch inspection:")
    print(f"  - Features batch shape: {features_batch.shape}")
    print(f"  - Labels batch shape: {labels_batch.shape}")
    print(f"  - Features dtype: {features_batch.dtype}")
    print(f"  - Labels dtype: {labels_batch.dtype}")
    print(f"  - Features min/max: {features_batch.min():.4f} / {features_batch.max():.4f}")
    print(f"  - Unique labels in batch: {labels_batch.unique().tolist()}")
    
    # Show how to convert label indices back to perturbation names
    if 'inverse_label_mapping' in metadata:
        print(f"\nLabel indices to perturbation names:")
        for label_idx in labels_batch.unique()[:5]:
            pert_name = metadata['inverse_label_mapping'][label_idx.item()]
            print(f"  {label_idx.item()} -> {pert_name}")
    
    return features_batch, labels_batch


def main():
    """
    Run all examples.
    """
    print("\n" + "🧬" * 30)
    print("NORMAN PERTURBATION DATA LOADING EXAMPLES")
    print("🧬" * 30)
    
    try:
        # Run examples
        example_basic_loading()
        train_loader, val_loader, test_loader, metadata = example_dataloader_creation()
        example_batch_inspection()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Use these dataloaders in train.py for model training")
        print("  2. Modify src/models/base_model.py to match your data dimensions")
        print("  3. Experiment with different batch sizes and splits")
        print("  4. Add data augmentation if needed")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPlease ensure:")
        print("  1. The perturbation data analysis module is available")
        print("  2. All dependencies are installed (see requirements.txt)")
        print("  3. The project structure is set up correctly")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check your data directory and module paths.")


if __name__ == "__main__":
    main()
