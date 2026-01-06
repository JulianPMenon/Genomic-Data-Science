"""
Training script for genomic ML models
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.models.base_model import BaseModel
from src.data.dataset import GenomicDataset
from src.data.dataloader import get_dataloaders
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics, print_metrics


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        logger: Logger object
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device, logger):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        logger: Logger object
        
    Returns:
        tuple: (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] == 2 else probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics


def train(config_path):
    """
    Main training function.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=log_dir / "training.log")
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device(
        f"cuda:{config['device']['gpu_id']}" 
        if config['device']['use_cuda'] and torch.cuda.is_available() 
        else "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['data']['num_workers'],
        seed=config['data']['seed']
    )
    
    # Create model
    logger.info("Initializing model...")
    model = BaseModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Scheduler
    if config['training']['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
    else:
        scheduler = None
    
    # Tensorboard
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, logger)
        logger.info(f"Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, logger)
        
        # Tensorboard logging
        if config['logging']['tensorboard']:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1_score'], epoch)
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / f"best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Test evaluation
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pth")['model_state_dict'])
    test_loss, test_metrics = validate(model, test_loader, criterion, device, logger)
    logger.info(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, logger)
    
    if config['logging']['tensorboard']:
        writer.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train genomic ML model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train(args.config)
