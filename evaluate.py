"""
Evaluation/Inference script for genomic ML models
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.models.base_model import BaseModel
from src.data.dataset import GenomicDataset
from src.data.dataloader import get_single_dataloader
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics, print_metrics


def evaluate(model, data_loader, criterion, device, logger):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        logger: Logger object
        
    Returns:
        tuple: (average loss, metrics dictionary, predictions)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
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
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    if len(set(all_labels)) == 2:
        # Binary classification
        probs_for_roc = [p[1] for p in all_probs]
        metrics = calculate_metrics(all_labels, all_preds, probs_for_roc, average='binary')
    else:
        # Multi-class classification
        metrics = calculate_metrics(all_labels, all_preds, all_probs, average='macro')
    
    return avg_loss, metrics, (all_preds, all_labels, all_probs)


def inference(model, data_loader, device, logger):
    """
    Run inference on unlabeled data.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run inference on
        logger: Logger object
        
    Returns:
        tuple: (predictions, probabilities)
    """
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for features, _ in tqdm(data_loader, desc="Inference"):
            features = features.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_probs


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """
    # Setup logger
    logger = setup_logger()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create minimal config for evaluation
        config = {
            'model': {
                'input_dim': 100,
                'hidden_dim': 256,
                'output_dim': 2,
                'dropout': 0.5
            },
            'device': {
                'use_cuda': True,
                'gpu_id': 0
            }
        }
    
    # Set device
    device = torch.device(
        f"cuda:{config['device']['gpu_id']}" 
        if config['device']['use_cuda'] and torch.cuda.is_available() 
        else "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = BaseModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    dataset = GenomicDataset(data_path=args.data_path)
    data_loader = get_single_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Run evaluation or inference
    if args.mode == 'evaluate':
        logger.info("Running evaluation...")
        criterion = nn.CrossEntropyLoss()
        loss, metrics, predictions = evaluate(model, data_loader, criterion, device, logger)
        
        logger.info(f"Average Loss: {loss:.4f}")
        print_metrics(metrics, logger)
        
        # Save predictions if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            preds, labels, probs = predictions
            np.savez(
                output_path,
                predictions=preds,
                labels=labels,
                probabilities=probs
            )
            logger.info(f"Predictions saved to {output_path}")
    
    else:  # inference mode
        logger.info("Running inference...")
        preds, probs = inference(model, data_loader, device, logger)
        
        logger.info(f"Processed {len(preds)} samples")
        
        # Save predictions
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                output_path,
                predictions=preds,
                probabilities=probs
            )
            logger.info(f"Predictions saved to {output_path}")
        else:
            logger.warning("No output path specified. Predictions not saved.")
    
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or run inference with genomic ML model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (CSV, etc.)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['evaluate', 'inference'],
        default='evaluate',
        help="Mode: 'evaluate' (with labels) or 'inference' (without labels)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation/inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions"
    )
    
    args = parser.parse_args()
    main(args)
